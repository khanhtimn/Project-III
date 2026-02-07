= Giới thiệu

Các Mô hình Ngôn ngữ Lớn (LLM) thể hiện năng lực học theo ngữ cảnh  ấn tượng, tận dụng thông tin từ đầu vào để đạt hiệu năng vượt trội trên nhiều bài toán hạ nguồn. Ví dụ, trong hội thoại nhiều lượt @roller2021recipes, @zhang2020dialogue, việc đưa lịch sử hội thoại vào ngữ cảnh giúp LLM phản hồi truy vấn của người dùng tốt hơn. Trong sinh văn bản tăng cường truy hồi (Retrieval-Augmented Generation -- RAG) @guu2020retrieval, @izacard2022few, LLM có thể sinh câu trả lời chính xác hơn bằng cách sử dụng các đoạn văn được truy hồi từ nguồn tri thức bên ngoài, như web hoặc cơ sở tri thức nội bộ.

Tuy nhiên, khi tăng độ dài prompt để khai thác ngữ cảnh, độ trễ suy luận và chi phí bộ nhớ đều tăng đáng kể @yen2024long. Cụ thể, prompt dài hơn yêu cầu thêm bộ nhớ cho bộ nhớ khoá–giá trị (KV cache), vốn tăng tuyến tính theo độ dài prompt. Hơn nữa, thời gian tới token đầu tiên (time-to-first-token, TTFT) tăng xấp xỉ bậc hai theo độ dài prompt, trong khi thời gian cho mỗi token tiếp theo (time-to-iterative-token, TTIT) tăng gần tuyến tính. Kết quả là thông lượng suy luận giảm khi ngữ cảnh dài, làm hạn chế khả năng áp dụng LLM trong các kịch bản đòi hỏi throughput cao và độ trễ thấp, như hệ thống tìm kiếm ở quy mô web hay các hệ thống agentic tương tác thời gian thực.

Tối ưu độ trễ suy luận cho LLM với ngữ cảnh rất dài hiện là vấn đề được nhiều chú ý, với nhiều tiếp cận khác nhau: thay đổi độ phức tạp của attention @beltagy2020longformer, làm thưa attention và ngữ cảnh @child2019generating, @xiao2024efficient, @jiang2024longllmlingua, hoặc thay đổi chiến lược đưa ngữ cảnh vào mô hình @yen2024long. Tuy nhiên, em cho rằng RAG là một trường hợp đặc biệt, cần được đối xử riêng thay vì nhìn như “LLM ngữ cảnh dài nói chung”.

== Đặc thù của ngữ cảnh trong RAG

Trong các hệ thống RAG, phần lớn ngữ cảnh mà LLM nhận được được tạo bằng cách nối nhiều đoạn truy hồi từ bộ nhớ ngoài (vector DB, công cụ tìm kiếm, v.v.). Khác với ngữ cảnh thuần văn bản liên tục (ví dụ như một chương sách), ngữ cảnh này có các đặc điểm rất đặc biệt mà nếu bỏ qua sẽ dẫn đến thiết kế hệ thống kém hiệu quả.


Có ba đặc điểm chính như sau:

+ *Phân bổ token kém hiệu quả.*

  Ngữ cảnh RAG thường “thưa thông tin”: nhiều đoạn truy hồi không hữu ích cho truy vấn hiện tại, hoặc bị lặp lại giữa các truy vấn khác nhau. Tuy nhiên, nếu cứ đưa toàn bộ token vào LLM, hệ thống vẫn phải cấp bộ nhớ KV cache và tính toán attention cho tất cả các token này. Điều này dẫn tới lãng phí tài nguyên: chi phí cho những token không giúp ích cho câu trả lời lại chiếm phần lớn ngân sách thời gian và bộ nhớ.
  
+ *Lãng phí thông tin đã được mã hoá trong pipeline truy hồi.*

  Trước khi đến LLM sinh, pipeline RAG đã thực hiện nhiều bước xử lý: chia tài liệu thành các đoạn (chunk), mã hoá chúng thành vector, tính độ tương tự với truy vấn, re-rank, khử trùng lặp, v.v. Nghĩa là hệ thống đã có sẵn các embedding về từng đoạn và mức độ liên quan của chúng với truy vấn. Tuy nhiên, khi sinh câu trả lời, thường lại bỏ qua toàn bộ embedding này và chỉ đưa lại nguyên văn token của đoạn vào LLM. Thông tin đã được mã hoá và sắp xếp trước đó bị “vứt đi” ở bước quan trọng nhất là decoding.

+ *Cấu trúc chú ý thưa và bất thường.*

  Do các bước đa dạng hoá, khử trùng lặp, và do bản chất khác nhau của các đoạn truy hồi, phần lớn các đoạn trong ngữ cảnh không liên quan trực tiếp với nhau. Phân tích các ma trận chú ý cho thấy attention giữa token thuộc cùng một đoạn thường lớn hơn rất nhiều so với attention giữa các đoạn khác nhau; ma trận chú ý có dạng gần "block-diagonal" @yen2024long. Điều này nghĩa là hầu hết attention chéo giữa các đoạn thực chất là gần như bằng 0, nhưng hệ thống vẫn phải trả chi phí tính toán đầy đủ.

Ba yếu tố này gợi ý rằng: _Nếu vẫn coi RAG như một bài toán LLM ngữ cảnh dài tổng quát, sẽ bỏ lỡ những tối ưu rất rõ ràng_. Ngược lại, nếu thiết kế cơ chế giải mã (decoding) “nhìn thẳng” vào cấu trúc của ngữ cảnh RAG (rời rạc, nhiều đoạn, attention thưa, đã có embedding sẵn), có thể cắt giảm một lượng lớn phép tính mà gần như không ảnh hưởng tới chất lượng mô hình.

== REFRAG: Giải mã có chọn lọc cho RAG

Để giải quyết những thách thức trên, các nhà nghiên cứu từ Meta Superintelligence Labs đã đề xuất REFRAG (REpresentation For RAG), một khung giải mã hiệu quả được thiết kế riêng cho các ứng dụng RAG. Thay vì đối xử với RAG như một bài toán _ngữ cảnh dài tổng quát_, REFRAG khai thác các đặc điểm riêng của ngữ cảnh RAG để tối ưu hóa hiệu suất suy luận.

Cách tiếp cận chính của REFRAG xoay quanh ba thay đổi quan trọng trong quá trình giải mã:

+ Thay vì đưa trực tiếp token của các đoạn truy hồi vào decoder, REFRAG tận dụng _embedding đoạn đã được nén_ – thường đã được tính sẵn trong pipeline truy hồi bằng encoder như RoBERTa.

+ Các chunk embedding này được chiếu vào không gian embedding token của decoder thông qua một projection layer và được đưa _trực tiếp_ vào LLM như thể đó là "token đặc biệt đại diện cho một đoạn".

+ Một chính sách học tăng cường (RL policy) gọn nhẹ được huấn luyện để quyết định đoạn nào nên mở rộng thành công token đầy đủ, đoạn nào chỉ cần giữ dạng embedding nén – cho phép điều chỉnh mức độ nén động theo thời điểm suy luận.

Cách thiết kế này mang lại ba lợi ích chính:

- *Rút ngắn chiều dài đầu vào của decoder:* số "vị trí chú ý" tỉ lệ với số đoạn (L) thay vì số token (s), giúp phân bổ ngân sách token hiệu quả hơn.
- *Tái sử dụng embedding đã có:* không cần đưa token qua LLM để encoder nội bộ mã hoá lại; dùng trực tiếp embedding từ encoder bên ngoài, tiết kiệm tính toán.
- *Giảm độ phức tạp chú ý:* chi phí attention giờ đây tăng bậc hai theo số đoạn, thay vì theo số token trong toàn bộ ngữ cảnh. Với mỗi đoạn dài $k$ token, có thể tiết kiệm tới xấp xỉ $k$ lần số vị trí.

Theo bài báo REFRAG, mô hình đạt được những kết quả ấn tượng:

- *$30.85 times$ tăng tốc thời gian đến token đầu tiên (TTFT)* (tương đương $3.75 times$ so với phương pháp trước đó CEPE) mà _không_ làm xấu đi perplexity;
- Cho phép *mở rộng hiệu quả cửa sổ ngữ cảnh lên $16 times$* so với LLM gốc, nhờ đưa embedding đoạn vào thay vì token đầy đủ;
- *$7 times$ tăng thông lượng suy luận* và có thể xử lý 80 đoạn với độ trễ tương đương với RAG tiêu chuẩn xử lý 10 đoạn.

== Mục tiêu của đề tài

Bài báo REFRAG trình bày một tiếp cận hoàn toàn mới cho bài toán suy luận nhanh với RAG. Mục tiêu của em trong đề tài này là:

+ *Tìm hiểu sâu kiến trúc của REFRAG*: Phân tích cách mô hình nén, chiếu, và mở rộng có chọn lọc các đoạn ngữ cảnh, cùng với vai trò của từng thành phần (encoder RoBERTa, projection layer, policy network).

+ *Nghiên cứu quy trình huấn luyện 4 giai đoạn*: Khám phá lý do tại sao reconstruction task là bước khởi động quan trọng, làm thế nào continual pre-training kết hợp với curriculum learning, và cách RL policy được huấn luyện để tối ưu hóa sự cân bằng giữa tốc độ và chất lượng.

+ *Đánh giá hiệu suất trên các bài toán thực tế*: Kiểm chứng lại các kết quả báo cáo trên các tập dữ liệu RAG, hội thoại nhiều lượt, và tóm tắt tài liệu dài, đồng thời khám phá những ưu điểm và hạn chế của mô hình.

+ *Xây dựng triển khai của REFRAG*: Cài đặt các thành phần chính của mô hình và tuning training pipeline để hiểu rõ hơn cách thức hoạt động và những yếu tố quyết định đến hiệu suất.

Các phần sau sẽ trình bày chi tiết kiến trúc của REFRAG, thanh lọc quy trình huấn luyện, cơ chế chọn lọc mở rộng bằng RL, cũng như các kết quả thực nghiệm và nhận xét về mô hình.