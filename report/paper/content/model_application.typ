= Các ứng dụng và triển khai

 Sau khi huấn luyện RAG ở mức mô hình ngôn ngữ, có thể áp dụng kiến trúc này cho nhiều ứng dụng học theo ngữ cảnh khác nhau. Mục tiêu là kiểm tra liệu các lợi ích về độ trễ và bộ nhớ có chuyển hoá thành _cải thiện hoặc ít nhất giữ nguyên_ chất lượng trên các tác vụ thực tế, chứ không chỉ dừng ở log-perplexity.

Trong chương này, em tập trung vào ba nhóm ứng dụng:
  +  Hệ thống hỏi–đáp tăng cường truy hồi (RAG) một lượt.
  +  Hội thoại nhiều lượt cần tri thức với RAG ở mỗi lượt.
  +  Các bài toán ngữ cảnh rất dài như tóm tắt tài liệu dài.

== Hệ thống RAG một lượt

=== Mục tiêu và bối cảnh

 Trong RAG một lượt, pipeline thường gồm:
  +  Nhận câu hỏi $q $ từ người dùng.
  +  Truy hồi K đoạn $C_(1), ..., C_(K)$ từ kho tri thức bằng retriever.
  +  Ghép $q $ và ${C_(i)}$ thành prompt rồi đưa vào LLM để sinh đáp án $a $.

 Vấn đề là khi K tăng (để tăng khả năng chứa thông tin liên quan), prompt trở nên rất dài, gây tăng TTFT và chi phí bộ nhớ. Trong các hệ thống thực tế, để giữ độ trễ chấp nhận được, người ta thường phải:
  -  cắt bớt số đoạn (K nhỏ),
  -  hoặc rút gọn từng đoạn bằng một LLM khác,
  -  hoặc chỉ giữ một phần của mỗi đoạn.

Mô hình RAG này cho phép _giữ K lớn_ nhưng vẫn không vượt quá ngân sách token trong decoder, nhờ nén mỗi đoạn thành embedding.

=== Thiết lập huấn luyện

Trong huấn luyện RAG một lượt, em sử dụng tập dữ liệu hỗn hợp như đã mô tả ở Chương 4:
  -  Các bộ QA mở và đọc hiểu (CommonsenseQA, WebQuestions, SQuADv2, DROP, PubMedQA, ...).
  -  Một số bộ có chain-of-thought để mô hình học cách suy luận (GSM8K, StrategyQA, Algebra QA with Rationales, ...).

 Pipeline huấn luyện:
  +  Với mỗi câu hỏi $q $, em truy hồi K đoạn $C_(1), ..., C_(K)$ từ kho tri thức bằng DRAGON+.
  +  Chuẩn bị input theo hai phiên bản:

    -  Baseline LLaMA-FT: ghép $q $ và các đoạn $C_(i)$ thành prompt dạng chuỗi token đầy đủ.
    -  Customized RAG: giữ $q $ dưới dạng token, còn mỗi đoạn $C_(i)$ được encoder mã hoá thành $c_(i)$ và chiếu thành $tilde(c)_(i)$.

  +  Huấn luyện theo kiểu instruction-tuning: mô hình sinh đáp án $a $ dạng tự do, hàm mất mát là cross-entropy trên $a $.

=== Thiết lập đánh giá

 Em đánh giá trên nhiều benchmark trong bộ KILT và các tập QA phổ biến:
  -  *Lựa chọn đáp án:* MMLU, BoolQ, SIQA, PIQA, HellaSwag, Winogrande.
  -  *QA thực tế:* TriviaQA, NaturalQuestions, FEVER.

Đối với các bài toán lựa chọn đáp án, đầu ra của mô hình được ánh xạ về một trong các lựa chọn (A, B, C, D); đối với QA mở, em đánh giá bằng exact-match hoặc F1 dựa trên token.

=== Kết quả và phân tích

 Với thiết lập “ngữ cảnh ngắn và cùng ngân sách token”:
  -  LLaMA-FT được phép dùng 1 đoạn truy hồi (K=1) trong prompt.
  -  Mô hình RAG$(8)$ dùng 8 đoạn, nhưng nén với $k = 8 $, sao cho tổng số token đi qua decoder xấp xỉ như LLaMA-FT.

 Các kết quả cho thấy:
  -  Trên phần lớn bài toán multiple-choice, Mô hình RAG$(8)$ _nhỉnh hơn_ hoặc tương đương LLaMA-FT, chứng tỏ việc nén không làm mất thông tin quan trọng ở mức nội dung.
  -  Trong thiết lập retriever yếu (chọn ngẫu nhiên từ 200 đoạn gần nhất), Mô hình RAG$(8)$ vượt LLaMA-FT rõ rệt, vì việc có nhiều đoạn (K=8) giúp tăng xác suất chứa bằng chứng liên quan, trong khi chi phí tăng rất ít nhờ nén.
  -  Mô hình RAG$(1 6)$ và Mô hình RAG$(3 2)$ giữ được hiệu năng cạnh tranh, cho thấy mô hình vẫn “chịu được” mức nén cao khi đã được CPT + tái dựng tốt.

 Trong thiết lập “ngữ cảnh dài” với K=40 hoặc K=80:
  -  LLaMA-FT không thể dùng hết 80 đoạn vì giới hạn cửa sổ 4k token, phải cắt bớt đoạn hoặc nội dung, nên chất lượng dừng lại sau một ngưỡng K.
  -  Customized RAG (nhất là với $k = 1 6,3 2 $) vẫn sử dụng được tới 80 đoạn, do phần context chủ yếu đi qua encoder và được nén.
  -  Kết quả trên các bài toán như TriviaQA và NQ cải thiện đáng kể khi tăng K từ 10 lên 40/80 trong Customized RAG, nhưng gần như không cải thiện trong LLaMA-FT (do không thể tận dụng thêm đoạn).

== Hội thoại nhiều lượt với RAG

=== Đặc điểm hội thoại nhiều lượt

 Trong hội thoại nhiều lượt, mô hình phải xử lý:
  -  Lịch sử hội thoại dài (có thể hàng chục lượt trao đổi).
  -  Ngữ cảnh truy hồi mới tại mỗi lượt (các đoạn văn liên quan đến câu hỏi hiện tại).

 Prompt cho mỗi lượt thường là: 
 
 [$"role: user/system"] space.nobreak "lịch sử hội thoại" + "câu hỏi hiện tại" + "các đoạn truy hồi"$

nên tổng số token tăng rất nhanh theo số lượt. Một LLM với cửa sổ 4k token dễ dàng bị tràn khi hội thoại kéo dài và K lớn.

=== Áp dụng

 Em áp dụng mô hình RAG theo cách:
  -  *Lịch sử hội thoại*: vẫn giữ dưới dạng token bình thường, vì độ dài thường nhỏ hơn phần context truy hồi tổng hợp từ nhiều lượt.
  -  *Ngữ cảnh truy hồi*: ở mỗi lượt, các đoạn $C_(1), ..., C_(K)$ được nén thành embedding đoạn bằng encoder, rồi đưa vào decoder.

Ở một số thiết lập, áp dụng nén chọn lọc dựa trên RL:
  -  Những đoạn có embedding cho thấy ít liên quan (theo chính sách) sẽ được nén.
  -  Một vài đoạn được mở rộng ra token đầy đủ nếu chính sách đánh giá là quan trọng.

=== Thiết lập đánh giá hội thoại

 Trên TopiOCQA, ORConvQA và QReCC, em thiết lập:
  -  Mỗi lượt hội thoại, câu hỏi hiện tại được truy hồi K đoạn từ kho tri thức (K=5 hoặc K=10).
  -  Chỉ xét các hội thoại có độ dài tối thiểu $L $ lượt (ví dụ $L in {2,4,6 }$) để phân tích hiệu ứng “dài dần”.
  - Đánh giá hiệu năng bằng độ chính xác hoặc F1 trên câu trả lời của mỗi lượt.

=== Kết quả

 Kết quả cho thấy:
  -  Khi số lượt hội thoại nhỏ (2–3), LLaMA-FT và Customized RAG có chất lượng tương đương, vì tổng số token vẫn trong giới hạn cửa sổ.
  -  Khi số lượt tăng lên (5–6 trở lên), LLaMA-FT bắt đầu phải cắt lịch sử hội thoại hoặc giảm số đoạn truy hồi, trong khi Customized RAG vẫn giữ được nhiều lịch sử và đoạn hơn nhờ nén.
  - Ở K=10, Customized RAG vượt LLaMA-FT trên cả TopiOCQA, ORConvQA, QReCC, đặc biệt ở các hội thoại dài; điều này cho thấy nén theo đoạn là chiến lược hiệu quả để duy trì “trí nhớ” hội thoại.

== Tóm tắt tài liệu dài và các tác vụ ngữ cảnh dài khác

=== Thách thức của tóm tắt tài liệu dài

 Tóm tắt tài liệu dài (bài báo khoa học, chương sách, báo cáo) là một bài toán kinh điển của ngữ cảnh dài:
  -  Tài liệu đầu vào có thể dài hơn 10k token.
  -  Mô hình cần nhìn được cấu trúc tổng thể, chứ không chỉ vài đoạn đầu/cuối.

 Các tiếp cận thông thường:
  -  Tóm tắt từng đoạn/bố cục rồi ghép (hierarchical summarization).
  -  Cắt bớt phần giữa (giữ đầu + cuối).
  -  Dùng LLM hỗ trợ context dài (32k, 64k), nhưng chi phí rất cao.

=== Áp dụng

 Mô hình RAG cho phép:
  -  Chia tài liệu thành các đoạn $C_(i)$ (ví dụ 256 token/đoạn).
  -  Mã hoá từng $C_(i)$ bằng encoder và đưa embedding đoạn $tilde(c)_(i)$ vào decoder.

Có thể triển khai hai chiến lược:
  +  *Nén toàn bộ:* tất cả đoạn đều được nén thành embedding, và decoder sinh bản tóm tắt dựa trên dãy embedding này. Chi phí thấp nhất, TTFT nhỏ, nhưng có nguy cơ mất chi tiết quan trọng.
  +  *Nén chọn lọc:* chính sách RL hoặc heuristic (ví dụ dựa trên độ dài, vị trí, embedding) quyết định mở rộng một số đoạn thành token đầy đủ (chẳng hạn phần mở đầu, kết luận, hoặc đoạn có độ bất thường cao). Phần còn lại vẫn giữ dạng embedding nén.

=== Kết quả định tính

 Kết quả định tính và một số thước đo tự động (ROUGE, BLEU) cho thấy:
  -  Customized RAG với nén chọn lọc tạo ra bản tóm tắt có chất lượng tương đương hoặc tốt hơn LLaMA-2-7B được tinh chỉnh tóm tắt, trong khi chi phí tính toán thấp hơn đáng kể.
  - Đặc biệt, việc mô hình có thể “nhìn” toàn bộ tài liệu (qua embedding đoạn) giúp bản tóm tắt giữ được nhiều thông tin toàn cục mà cách tóm tắt đoạn-lẻ khó đạt được.
  -  Các bản tóm tắt của Customized RAG ít bị lặp ý và phân đoạn “đứt gãy” hơn so với việc tóm tắt từng đoạn rồi ghép.

== Thảo luận và hạn chế

 Mặc dù mô hình RAG cho kết quả tích cực trên nhiều tác vụ, cũng ghi nhận một số hạn chế:
  -  *Phụ thuộc vào encoder:* nếu encoder $M_(upright(e n c))$ và tầng chiếu $phi.alt $ không được huấn luyện tốt (ví dụ bỏ nhiệm vụ tái dựng), chất lượng embedding đoạn sẽ kém, dẫn đến mất mát thông tin khi nén.
  -  *Chi phí triển khai pipeline:* hệ thống thực tế cần tích hợp thêm encoder cho context, caching embedding, và chính sách RL, làm pipeline phức tạp hơn so với RAG thuần token.
  -  *Tương tác phức tạp giữa token và embedding:* khi vừa có token vừa có embedding trong cùng chuỗi (nén chọn lọc), mô hình cần thời gian huấn luyện đủ lâu để học cách “trộn” hai dạng biểu diễn này hiệu quả.

 Tuy vậy, các kết quả cho thấy Customized RAG là một hướng khả thi để _giảm độ trễ_ và _mở rộng ngữ cảnh_ mà vẫn _bảo toàn hoặc cải thiện_ chất lượng trên các tác vụ thực tế.
