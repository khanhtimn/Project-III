= Công trình liên quan và Kết luận

 Trong chương này, em đặt bài báo REFRAG vào bối cảnh các nghiên cứu trước đây về LLM ngữ cảnh dài, RAG, nén ngữ cảnh và RL cho tối ưu tổ hợp. Mục tiêu của phần này là tổng hợp và làm rõ ý tưởng cốt lõi của REFRAG, những điểm khác biệt so với các hướng liên quan, và rút ra các hướng phát triển tiếp theo.

== Mô hình ngôn ngữ tăng cường truy hồi

 Mô hình ngôn ngữ tăng cường truy hồi là một chủ đề đã được nghiên cứu rộng rãi:
  -  Guu et al. @guu2020retrieval giới thiệu tiền huấn luyện với truy hồi cho masked LM, cho thấy truy hồi có thể cải thiện hiệu năng trên nhiều nhiệm vụ.
  -  Borgeaud et al. @borgeaud2022trillions đề xuất kiến trúc LLM sinh đọc từ một kho tri thức khổng lồ qua cross-attention, huấn luyện end-to-end trên lượng dữ liệu truy hồi ở quy mô rất lớn.
  -  REPLUG @shi2024replug và RA-DIT @lin2024radit xem RAG như một lớp “plug-in” nằm ngoài, tập trung vào cách dùng lại LLM nền bằng cách bổ sung đoạn truy hồi vào prompt và tinh chỉnh theo hướng dual-instruction.
  -  Fusion-in-Decoder (FiD) @izacard2021fid sử dụng encoder để xử lý từng đoạn độc lập, sau đó ghép các trạng thái ẩn vào decoder; cách này giảm phần nào chi phí do đoạn được xử lý song song, nhưng vẫn dùng biểu diễn dài cho decoder.

 So với các cách tiếp cận này, REFRAG nhấn mạnh:
  -  Tập trung vào *giai đoạn giải mã* (decoding) và chi phí chạy, thay vì chỉ tập trung vào chất lượng dưới ngữ cảnh đầy đủ.
  -  *Tái sử dụng embedding* từ pipeline truy hồi (thay vì luôn cần tokenizer + embedding nội bộ của LLM cho toàn bộ ngữ cảnh).
  -  Cho phép *nén ngữ cảnh ở mức đoạn* một cách có hệ thống, và hỗ trợ nén ở vị trí bất kỳ, không chỉ prefix.

== LLM ngữ cảnh dài và attention hiệu quả

 Rất nhiều nghiên cứu tập trung vào việc làm LLM xử lý ngữ cảnh dài hiệu quả hơn:
  -  Sparse / local attention như Longformer @beltagy2020longformer và Sparse Transformer @child2019generating.
  -  Attention gần đúng tuyến tính như Performer @choromanski2021performer.
  -  Kỹ thuật quản lý KV cache như StreamingLLM @xiao2024efficient, giúp xử lý “dòng” token dài bằng cách dùng một số token làm “attention sink”.
  -  CEPE @yen2024long dùng context encoder và cross-attention để đọc từ embedding ngữ cảnh, linh hoạt hơn LLM thuần self-attention.

 Các công trình này chủ yếu thay đổi:
  -  kiến trúc attention,
  -  hoặc cách lưu/truy cập KV cache.
 Có thể xem REFRAG là một bổ sung theo “chiều” khác:
  -  _Không thay_ kiến trúc attention của decoder; thay vào đó, thay đổi _biểu diễn ngữ cảnh_ đưa vào decoder bằng cách nén thành embedding đoạn.
  -  Phần lớn chi phí của ngữ cảnh được chuyển sang encoder (có thể pre-compute, song song, cache), trong khi decoder thấy một chuỗi ngắn hơn nhiều.

 Trong tương lai, việc kết hợp REFRAG với các kỹ thuật attention hiệu quả (ví dụ Performer + biểu diễn embedding đoạn) có thể giúp tăng thêm mức tiết kiệm chi phí.

== Compressive transformer và bộ nhớ embedding

 Compressive Transformer @rae2020compressive và các công trình sau @chevalier2023compress @dai2025pcc @kuratov2025cramming nghiên cứu cách:
  -  Nén KV cache để giảm bộ nhớ,
  -  Tóm tắt ngữ cảnh thành một số embedding ngắn hơn, dùng như bộ nhớ lâu dài.

 Các công trình này cho thấy:
  -  Có thể nén hàng nghìn token vào một embedding duy nhất (với chiều đủ lớn) mà vẫn bảo toàn nhiều thông tin có ích.
  -  Việc huấn luyện mô hình để “đọc” lại từ bộ nhớ nén là khả thi nếu có mục tiêu và giáo trình phù hợp.

 REFRAG có cùng tinh thần với các hướng “bộ nhớ embedding”:
  -  Mỗi đoạn $C_(i)$ được nén thành embedding $c_(i)$, rồi chiếu thành $tilde(c)_(i)$ để decoder sử dụng.
  -  Nhiệm vụ tái dựng và CPT với giáo trình giúp mô hình học cách sử dụng embedding này hiệu quả.
 Tuy nhiên, REFRAG tập trung vào một bối cảnh cụ thể hơn:
  -  Tập trung vào _ngữ cảnh kiểu RAG_, nơi các đoạn đã có embedding từ retriever trước đó.
  -  Hỗ trợ nén chọn lọc và khả năng _mở rộng lại_ đoạn thành token đầy đủ khi cần.

== Nén prompt và chọn lọc ngữ cảnh

 Nén prompt (prompt compression) là một hướng phổ biến khác:
  -  LLMLingua @jiang2023llmlingua và LongLLMLingua @jiang2024longllmlingua nén prompt bằng cách loại bỏ token ít quan trọng theo một thước đo dựa trên LLM, cho phép giảm độ dài prompt với mất mát nhỏ về chất lượng.
  -  Các phương pháp khác @li2023compress @liskavets2024prompt dùng encoder câu để xếp hạng câu và giữ lại các câu quan trọng, bỏ bớt phần ít liên quan.

 Các phương pháp nén prompt hoạt động ở _mức token hoặc câu_, trong khi REFRAG hoạt động ở _mức đoạn_. Chúng có thể bổ trợ lẫn nhau:
  -  Nén prompt trước ở mức câu/token bằng LLMLingua.
  -  Sau đó, nén thêm ở mức đoạn bằng REFRAG (Customized RAG trong phần thực nghiệm), đặc biệt đối với các đoạn truy hồi dài.

 Ngoài ra, nhiều hệ thống RAG dùng heuristic cứng để giới hạn số đoạn (ví dụ luôn lấy 5 đoạn gần nhất). REFRAG, với chính sách RL, cho thấy việc chọn đoạn để nén/mở rộng có thể được tối ưu hóa một cách _học được_, thay vì chỉ dùng luật cố định.

== Tối ưu tổ hợp và RL cho lựa chọn đoạn

 Việc chọn tập con đoạn để mở rộng (trong nén chọn lọc) là một bài toán tối ưu tổ hợp, có không gian tìm kiếm kích thước $binom(L, T')$, khó giải bằng các phương pháp gradient tiêu chuẩn. Các công trình trong tối ưu tổ hợp bằng mạng nơ-ron gợi ý dùng:
  -  Policy gradient (REINFORCE) hoặc các biến thể (PPO) để học chính sách chọn từng phần tử một cách tuần tự.
  -  Các kiến trúc như pointer network hoặc Transformer trên tập đối tượng để sinh ra phân phối trên tổ hợp.

REFRAG áp dụng ý tưởng này bằng cách:
  -  Dùng một transformer nhỏ trên embedding đoạn ${c_(i)}$ để sinh logit $s_(i)$ cho mỗi đoạn.
  -  Chọn lần lượt $T '$ đoạn theo phân phối softmax với mask, đảm bảo không chọn trùng.
  -  Dùng reward là âm log-perplexity trên đoạn đầu ra để cập nhật chính sách.

 Kết quả thực nghiệm cho thấy chính sách RL này vượt trội so với các heuristic như chọn ngẫu nhiên, chọn đoạn có perplexity thấp hoặc cao, củng cố luận điểm rằng:
  -  Việc nén chọn lọc ngữ cảnh nên được học, thay vì chỉ thiết kế bằng tay.

== Kết luận

 Trong bài báo cáo này, em chủ yếu đi theo bài báo REFRAG để tìm hiểu một khung giải mã hiệu quả cho các hệ thống sinh văn bản tăng cường truy hồi. Khác với nhiều nghiên cứu tập trung vào kiến trúc attention hoặc cách lưu KV cache, REFRAG tập trung vào:
  -  Cách biểu diễn ngữ cảnh RAG _bên trong_ LLM,
  -  Và cách sử dụng embedding đoạn để nén ngữ cảnh mà vẫn giữ được chất lượng.

 Theo bài báo, các đóng góp chính của REFRAG gồm:
  +  Một cơ chế giải mã dựa trên embedding đoạn cho ngữ cảnh RAG, cho phép nén ngữ cảnh ở mức đoạn và hỗ trợ nén ở vị trí bất kỳ trong chuỗi.
  +  Một quy trình huấn luyện (tái dựng, tiếp tục tiền huấn luyện (CPT) và học theo giáo trình) nhằm giúp decoder học cách sử dụng embedding đoạn như một dạng “token nén”.
  +  Nén chọn lọc bằng RL – một chính sách nhẹ giúp quyết định đoạn nào cần mở rộng thành token đầy đủ, đoạn nào chỉ cần embedding, tối ưu theo reward dựa trên log-perplexity hoặc hiệu năng tác vụ.
  +  Một đánh giá thực nghiệm rộng trên mô hình ngôn ngữ, RAG, hội thoại nhiều lượt và tóm tắt tài liệu dài. Theo các kết quả báo cáo, REFRAG:

    - đạt tới $30.85x$ tăng tốc TTFT so với LLaMA-2-7B và $3.75x$ so với CEPE trong một số cấu hình,
    -  giảm đáng kể kích thước KV cache và cải thiện throughput,
    -  giữ nguyên hoặc cải thiện độ chính xác trên nhiều bài toán ngữ cảnh dài.

== Hướng phát triển tiếp theo

 Dựa trên các quan sát từ REFRAG và các hướng liên quan, em thấy một số hướng mở hấp dẫn:
  -  *Kết hợp với attention hiệu quả:* Tích hợp REFRAG với các mô hình attention hiệu quả (Performer, Longformer, ...) để tối ưu cả _kiến trúc_ lẫn _biểu diễn ngữ cảnh_.
  -  *Mở rộng sang đa phương thức:* Áp dụng ý tưởng embedding đoạn cho các loại ngữ cảnh khác như hình ảnh, bảng, sơ đồ, trong đó mỗi “đoạn” có thể là một vùng ảnh hoặc một bảng con.
  -  *Tối ưu end-to-end toàn pipeline RAG:* Hiện tại, retriever, encoder, chính sách RL và decoder được huấn luyện tương đối tách rời; một hướng tiếp theo là huấn luyện end-to-end để tối ưu trực tiếp chất lượng và độ trễ toàn hệ thống.
  -  *Nghiên cứu sâu hơn về giới hạn nén:* Khảo sát chi tiết hơn mối quan hệ giữa $k $, kích thước embedding, độ sâu encoder và chất lượng; xác định “vùng an toàn” cho các cấu hình khác nhau.

 Tóm lại, REFRAG gợi ý rằng việc đối xử riêng với ngữ cảnh kiểu RAG – vốn có cấu trúc rất khác ngữ cảnh thuần văn bản liên tục – có thể mở ra nhiều hướng tối ưu hóa mới: vừa giữ được chất lượng, vừa cải thiện đáng kể hiệu năng hệ thống trong các ứng dụng thực tế.
