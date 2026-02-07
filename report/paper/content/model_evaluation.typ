= Kết quả thực nghiệm

 Trong phần này, em trình bày các thực nghiệm để đánh giá mô hình REFRAG từ nhiều góc độ:
  - Chất lượng mô hình ngôn ngữ (log-perplexity) trên văn bản dài.
  - Độ trễ suy luận: TTFT, TTIT, throughput, bộ nhớ KV cache.
  - Hiệu quả trên các tác vụ hạ nguồn: RAG, hội thoại nhiều lượt, tóm tắt tài liệu dài.
  - Các phân tích ablation về hệ số nén $k $, nhiệm vụ tái dựng, giáo trình và chính sách RL.

== Thiết lập thực nghiệm

=== Mô hình và phần cứng
  -  *Decoder*: Llama-3.2-3B (3B tham số, 28 tầng, 3072 chiều ẩn).
  -  *Encoder*: một mô hình encoder tương thích với pipeline RAG, ví dụ RoBERTa-large, khoảng 300–400M tham số.
  -  *Tầng chiếu* $phi.alt $: một lớp tuyến tính từ chiều embedding của encoder sang 3072.

 Hệ số nén $k $ khảo sát gồm: $  k in {8, 1 6, 3 2 },  $ tương ứng với việc mỗi chunk embedding đại diện cho 8, 16, hoặc 32 token ngữ cảnh.

 Các thí nghiệm về độ trễ được thực hiện trên GPU A100 80GB, với batch size 32 tuỳ thiết lập; TTFT, TTIT được đo bằng cách trung bình hoá trên nhiều lần chạy và lấy trung vị để giảm nhiễu.

=== Dữ liệu cho mô hình ngôn ngữ

 Dữ liệu huấn luyện CPT:
  -  Tập SlimPajama @soboleva2023slimpajama, chỉ dùng hai miền Book và ArXiv.
  -  Tổng khoảng 20B token, trong đó 50% từ sách, 50% từ bài báo khoa học.

 Dữ liệu đánh giá LM:

  -  SlimPajama-Book (tập test),
  -  SlimPajama-ArXiv (tập test),
  -  PG19 @rae2019pg19,
  -  ProofPile @azerbayev2023proofpile.
 Mỗi mẫu được cắt/tạo thành chuỗi $T = 4 0 9 6 $ token, với $s = 2 0 4 8 $ token ngữ cảnh và $o in {5 1 2, 1 0 2 4, 2 0 4 8 }$ token đầu ra để tính log-perplexity.

=== Dữ liệu cho RAG và hội thoại

 Cho RAG và hội thoại, em làm theo thiết lập trong @lin2024radit:
  -  *Đối thoại:* OpenAssistant Conversations.
  -  *QA mở:* CommonsenseQA, MathQA, WebQuestions, WikiQA, Yahoo! Answers, FreebaseQA, MS MARCO.
  -  *Đọc hiểu:* DROP, PubMedQA, QuaRel, SQuADv2.
  -  *Chuỗi suy luận:* Algebra QA with Rationales, Explanations for CommonsenseQA, GSM8K @shi2024deepseekmath, MathQA, StrategyQA.
 Tổng cộng khoảng 1.1M cặp dữ liệu dùng cho instruction-tuning.

 Dữ liệu đánh giá RAG:
  -  MMLU @hendrycks2021mmlu, BoolQ @clark2019boolq, SIQA @sap2019socialiqa, PIQA @bisk2020piqa.
  -  Các nhiệm vụ thuộc bộ KILT @petroni2020kilt: HellaSwag, Winogrande, TriviaQA, FEVER, NaturalQuestions.

 Dữ liệu hội thoại nhiều lượt với RAG:
  -  TopiOCQA @adlakha2022topiocqa,
  -  ORConvQA @qu2020orconvqa,
  -  QReCC @anantha2021qrecc.

=== Kho tri thức và truy hồi

 Kho tri thức dùng chung cho RAG và hội thoại được xây dựng từ:
  -  Dump Wikipedia,
  -  Tập con của CommonCrawl (hoặc web corpus tương tự).
 Các tài liệu được chia thành khoảng 400M đoạn (chunk), mỗi đoạn tối đa 200 từ. Mô hình truy hồi là DRAGON+ @lin2023dragon, được huấn luyện theo @izacard2023atlas, sử dụng cosine similarity trong không gian embedding để tìm K láng giềng gần nhất.

 Em khai báo hai chế độ:

  -  *Retriever mạnh*: Lấy trực tiếp K đoạn gần nhất từ DRAGON+.
  -  *Retriever yếu*: Lấy 200 đoạn gần nhất, rồi chọn ngẫu nhiên K đoạn để đưa vào mô hình sinh (mô phỏng tình huống pipeline thực tế bị nhiễu hoặc có lỗi).

=== Kết quả mô hình ngôn ngữ

==== So sánh với LLaMA và CEPE

 Em so sánh các mô hình sau:

  -  LLaMA-NoContext: không dùng ngữ cảnh $x_(1 : s)$.
  -  LLaMA-FullContext: dùng toàn bộ $x_(1 : T)$.
  -  LLaMA-32K: phiên bản hỗ trợ 32k token.
  -  LLaMA$(K)$: chỉ dùng $K $ token cuối trong $x_(1 : s)$ (ví dụ $K = 2 5 6 $).
  -  REPLUG @shi2024replug.
  -  CEPE @yen2024long.
  -  Customized RAG$(k)$, với $k in {8,1 6,3 2 }$.

 Kết quả log-perplexity (trung bình trên 4 tập SlimPajama-Book, SlimPajama-ArXiv, PG19, ProofPile) cho thấy:

  -  Customized RAG$(8)$ *tốt hơn CEPE* đáng kể trên cả 4 tập, dù có hệ số nén tương đương hoặc mạnh hơn, đồng thời độ trễ TTFT thấp hơn.
  -  Customized RAG$(1 6)$ gần như đạt được log-perplexity của Customized RAG$(8)$, nhưng nén nhiều hơn (ít vị trí ngữ cảnh hơn), giảm thêm chi phí cho decoder.
  -  Customized RAG$(3 2)$ có log-perplexity tương đương hoặc chỉ tệ hơn chút ít so với CEPE, nhưng có lợi thế rất lớn về độ trễ.
  -  Customized RAG$(8)$ tốt hơn LLaMA$(2 5 6)$ (dùng 256 token cuối) một khoảng đáng kể, trong khi hai mô hình có “số vị trí ngữ cảnh” tương tự. Điều này chứng tỏ embedding đoạn nén có khả năng chứa nhiều thông tin hơn so với việc chỉ cắt ngắn chuỗi.

 LLaMA-FullContext và LLaMA-32K, như dự đoán, cho log-perplexity thấp nhất (không nén), nhưng độ trễ và chi phí bộ nhớ rất cao, khó áp dụng trong hệ thống thực tế. Customized RAG giúp tiến gần chất lượng của các mô hình này với chi phí thấp hơn nhiều.

==== Mở rộng ngữ cảnh

 Trong một loạt thí nghiệm khác, em tăng độ dài ngữ cảnh $s in {4 0 9 6, 8 1 9 2, 1 6 3 8 4 }$, vượt cửa sổ 4k token của LLaMA-2-7B. Đối với LLaMA gốc, đây là việc khá khó do positional encoding và KV cache đều bị giới hạn; trong khi với Customized RAG, em:

  -  Giữ nguyên decoder (không sửa positional encoding),
  -  Chỉ tăng số đoạn và embedding đoạn ở đầu vào decoder.

 Kết quả log-perplexity cho thấy:

  -  Customized RAG$(8)$ và Customized RAG$(1 6)$ giữ được hoặc cải thiện log-perplexity khi $s $ tăng lên 4096, 8192 và 16384 trên SlimPajama-Book và ArXiv.
  - Điều này cho thấy mô hình thực sự sử dụng thêm thông tin từ ngữ cảnh, chứ không chỉ “đọc cho có”.
  -  LLaMA-32K (hoặc các mô hình tương đương) khi được tinh chỉnh để nhận 16k context vẫn có log-perplexity tốt, nhưng độ trễ gấp nhiều lần và cần nhiều tài nguyên hơn.

=== Độ trễ suy luận và bộ nhớ

 Em đo TTFT, TTIT, throughput và kích thước KV cache cho các thiết lập khác nhau. Các quan sát chính:

==== Time-to-first-token (TTFT)

 Với $s = 1 6 3 8 4 $ (ngữ cảnh trung–dài) và xuất $o = 1 2 8 $ token, em đo TTFT cho:
  -  LLaMA-2-7B gốc,
  -  CEPE,
  -  Customized RAG$(8)$, Customized RAG$(1 6)$, Customized RAG$(3 2)$.

 Kết quả:
  -  Customized RAG$(1 6)$ có thể đạt khoảng $1 6 . 5 times $ tăng tốc TTFT so với LLaMA-2-7B, và khoảng $2 . 1 times $ so với CEPE.
  -  Customized RAG$(3 2)$ có thể đạt tới khoảng $3 0 . 8 5 times $ tăng tốc TTFT so với LLaMA và $3 . 7 5 times $ so với CEPE, với log-perplexity vẫn tương đương.

==== Time-to-iterative-token (TTIT) và throughput

 Trong giai đoạn sinh, Customized RAG cũng giảm số vị trí trong KV cache từ $s $ xuống $s /k $, giúp:

  -  Giảm TTIT do lượng byte phải tải mỗi bước từ HBM giảm.
  -  Tăng throughput vì số token/sinh trên đơn vị thời gian không còn bị nghẽn bởi KV cache dài.

 Thực nghiệm với batch size lớn cho thấy Customized RAG cho throughput cao hơn LLaMA-2-7B khoảng $4 $–$6 . 8 times $, tuỳ hệ số nén $k $ và độ dài output $o $.

==== Bộ nhớ KV cache

 Như phân tích ở Chương 2, bộ nhớ KV cache giảm từ: $  4 d l b(s + o) "xuống" 4 d l b(s/k + o),  $ đem lại giảm đáng kể khi $s gt.double o $. Điều này cho phép:

  -  Chạy mô hình với ngữ cảnh dài hơn trên cùng một GPU.
  -  Tăng batch size ở chế độ sinh, cải thiện throughput trong hệ thống phục vụ nhiều yêu cầu song song.

=== Customized RAG cho RAG (QA mở)

==== Thiết lập ngữ cảnh ngắn

 Trong thiết lập “ngữ cảnh ngắn với cùng độ trễ”, em:

  -  Cho LLaMA-FT (LLaMA-2-7B đã instruction-tuning) dùng đúng _một_ đoạn truy hồi trong prompt.
  -  Cho Customized RAG$(8)$ dùng _tám_ đoạn truy hồi, nhưng nén với $k = 8 $, sao cho tổng số token vào decoder xấp xỉ như LLaMA-FT dùng 1 đoạn.

 Trên 16 tác vụ RAG (MMLU, BoolQ, SIQA, PIQA, HellaSwag, Winogrande, TriviaQA, FEVER, NQ, ...), em thấy:

  -  Customized RAG$(8)$ đạt độ chính xác trung bình tương đương hoặc cao hơn LLaMA-FT trong retriever mạnh.
  -  Với retriever yếu, Customized RAG$(8)$ vượt LLaMA-FT rõ rệt vì khả năng “đọc” thêm nhiều đoạn giúp tăng xác suất chứa thông tin liên quan.
  -  Customized RAG$(1 6)$ và Customized RAG$(3 2)$ cũng cho kết quả cạnh tranh, đặc biệt trên các tác vụ lựa chọn đáp án, cho thấy việc nén mạnh hơn không phá vỡ quá trình suy luận.

==== Thiết lập ngữ cảnh dài

 Trong thiết lập “ngữ cảnh dài”, em cho phép:

  -  LLaMA-FT dùng tối đa 10–15 đoạn (bị giới hạn bởi cửa sổ 4k token).
  -  Customized RAG sử dụng tới 80 đoạn truy hồi (khi $k = 8 $ hoặc 16) mà vẫn nằm trong ngân sách độ trễ hợp lý.

 Kết quả:

  -  LLaMA-FT nhanh chóng hết chỗ trong prompt khi số đoạn nhiều, phải cắt bớt đoạn hoặc tóm tắt, dẫn đến chất lượng không còn tăng khi thêm đoạn.
  -  Customized RAG tiếp tục hưởng lợi khi thêm đoạn mới, đặc biệt trong retriever yếu, vì luôn có xác suất cao hơn để ít nhất một trong số 80 đoạn chứa thông tin cần thiết.
  -  Customized RAG$(8)$ và Customized RAG$(1 6)$ ở 80 đoạn thường vượt LLaMA-FT ở 10–15 đoạn trên hầu hết tác vụ RAG.

=== Hội thoại nhiều lượt với RAG

==== Thiết lập hội thoại

 Trên TopiOCQA, ORConvQA và QReCC:

  -  Mỗi lượt hội thoại, câu hỏi hiện tại được truy hồi K đoạn.
  -  Prompt cho LLaMA-FT gồm lịch sử hội thoại (văn bản các lượt trước) + K đoạn truy hồi dưới dạng token.
  -  Trong Customized RAG, lịch sử hội thoại vẫn là token, nhưng K đoạn truy hồi được nén thành embedding đoạn (hoặc một phần mở rộng theo nén chọn lọc).

 Em đánh giá theo số lượt tối thiểu từ 2 đến 6 (tức chỉ xét các ví dụ có ít nhất $L $ lượt để phân tích hiệu ứng hội thoại dài).

==== Kết quả

 Khi số lượt hội thoại tăng:

  -  LLaMA-FT phải cắt bớt lịch sử hoặc giảm số đoạn truy hồi đưa vào prompt để không vượt quá 4k token. Việc cắt này làm mô hình “quên” thông tin quan trọng từ đầu hội thoại, làm giảm độ chính xác ở các lượt sau.
  -  Customized RAG, nhờ nén ngữ cảnh truy hồi, có thể giữ lại nhiều lịch sử hơn trong cùng cửa sổ vị trí. Các đoạn truy hồi được “gói” vào embedding đoạn, tiết kiệm rất nhiều token.

 Với K = 5 và K = 10 đoạn truy hồi mỗi lượt:

  -  Customized RAG vượt LLaMA-FT trên 2/3 bộ dữ liệu ở K=5.
  -  Customized RAG vượt LLaMA-FT trên 3/3 bộ dữ liệu ở K=10, đặc biệt rõ ở QReCC (chuỗi hội thoại dài, nhiều lượt).

=== Tóm tắt tài liệu dài

 Trong tóm tắt tài liệu dài, đầu vào có thể là một bài báo hoặc chương sách dài vượt xa 4k token. Một số baseline:

  -  Tóm tắt từng đoạn nhỏ rồi ghép.
  -  Cắt bớt phần giữa, chỉ giữ phần đầu và cuối.
  -  Sử dụng LLaMA-32K (nếu tài nguyên cho phép).

 Với Customized RAG, em:

  -  Chia tài liệu thành các đoạn $C_(i)$, mã hoá thành embedding đoạn $c_(i)$, chiếu thành $tilde(c)_(i)$.
  -  Tạo chuỗi đầu vào cho decoder gồm một prompt tóm tắt ngắn + chuỗi embedding đoạn (và một số đoạn được mở rộng nếu cần).

 Kết quả:

  -  Customized RAG cho chất lượng tóm tắt (F1, ROUGE, hoặc đánh giá human) ngang ngửa hoặc tốt hơn LLaMA-2-7B được tinh chỉnh tóm tắt, trong khi độ trễ thấp hơn.
  -  So với việc tóm tắt từng đoạn nhỏ rồi ghép, Customized RAG giữ được tính nhất quán toàn cục tốt hơn vì mô hình “nhìn” được toàn bộ tài liệu thông qua embedding đoạn.

=== Phân tích ablation

==== Ảnh hưởng của hệ số nén $k$

 Em so sánh $k in {8,1 6,3 2 }$:

  - $k = 8 $: log-perplexity tốt nhất, độ trễ vẫn được cải thiện rõ rệt so với LLaMA gốc.
  - $k = 1 6 $: compromis tốt giữa chất lượng và độ trễ; đa số kết quả LM và RAG chỉ kém $k = 8 $ một chút nhưng TTFT nhanh hơn và KV cache nhỏ hơn.
  - $k = 3 2 $: log-perplexity bắt đầu suy giảm rõ ràng trên một số tập, nhưng vẫn chấp nhận được cho nhiều tác vụ RAG, trong khi độ trễ được cải thiện rất mạnh.

==== Ảnh hưởng của nhiệm vụ tái dựng và giáo trình

 Em tiến hành các biến thể:

  -  CPT _có_ tái dựng + _có_ giáo trình,
  -  CPT _có_ tái dựng + _không_ giáo trình,
  -  CPT _không_ tái dựng + _có_ giáo trình,
  -  CPT _không_ tái dựng + _không_ giáo trình.

 Kết quả:

  -  Bỏ tái dựng làm log-perplexity xấu đi đáng kể trên PG19 và ProofPile, cho thấy encoder + $phi.alt $ không học được cách nén thông tin tốt.
  -  Bỏ giáo trình khiến quá trình huấn luyện không ổn định, đặc biệt khi $k = 1 6 $ hoặc 32, và log-perplexity gần như không cải thiện so với baseline.
  -  Kết hợp cả tái dựng và giáo trình là cần thiết để đạt hiệu năng tốt như trong các bảng kết quả chính.

==== So sánh RL với heuristic

 Như đã trình bày ở Chương 3, em so sánh:

  -  RL (Customized RAG-RL),
  -  Perplexity-asc,
  -  Perplexity-desc,
  -  Random.

 Khi thay đổi $p $ (tỉ lệ mở rộng) từ 0.1 đến 0.9, em quan sát:

  -  RL luôn đạt log-perplexity thấp nhất (tức tốt nhất) trên PG19 và ProofPile.
  -  Perplexity-desc tốt hơn Random, trong khi Perplexity-asc có hiệu năng không ổn định (một số dataset tốt, một số dataset xấu).
  -  Với RAG trên các tác vụ QA, RL cho độ chính xác cao hơn 1–3 điểm phần trăm so với heuristic, đặc biệt khi $p $ nhỏ (tức nén mạnh, ít đoạn được mở rộng).

 Những kết quả này củng cố luận điểm rằng:

  -  Việc nén ngữ cảnh RAG có thể được tối ưu hoá một cách có hệ thống bằng RL,
  -  Thay vì chỉ dựa vào các heuristic đơn giản như cắt bớt đoạn theo độ dài hoặc theo perplexity.
