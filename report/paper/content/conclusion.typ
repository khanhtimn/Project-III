= KẾT LUẬN

== Tổng kết nghiên cứu

Đồ án đã thực hiện nghiên cứu và triển khai mô hình PhoBERT cho bài toán phân loại cảm xúc văn bản tiếng Việt trên mạng xã hội. Nghiên cứu được thực hiện theo phương pháp khoa học có hệ thống, từ xây dựng nền tảng lý thuyết đến triển khai thực nghiệm và đánh giá kết quả.

== Mức độ hoàn thành mục tiêu

=== Phạm vi nghiên cứu đã thực hiện

Dữ liệu và ngôn ngữ: Sử dụng bộ dữ liệu UIT-VSMEC với 6.927 câu được gán nhãn theo 7 lớp cảm xúc, xử lý các đặc thù ngữ pháp đơn lập của tiếng Việt.
Mô hình và phương pháp: Triển khai PhoBERT-base với các kỹ thuật fine-tuning bao gồm learning rate scheduling, gradient accumulation, mixed precision training và early stopping.
Đánh giá: Áp dụng các chỉ số accuracy, precision, recall, macro-F1, weighted-F1 và confusion matrix với phân tích chi tiết.

#figure(
  table(
    columns: 4,
    table.header(
      [*Yêu cầu*], [*Mục tiêu*], [*Kết quả đạt được*], [*Đánh giá*]
    ),
    [Test Accuracy], [≥ 85% (mục tiêu 90%)], [63.20%], [Chưa đạt],
    [Macro-F1], [≥ 75%], [60.07%], [Chưa đạt],
    [Weighted-F1], [≥ 80%], [63.18%], [Chưa đạt],
    [Kiến trúc], [PhoBERT], [✓ PhoBERT-base], [Đạt],
    [Kỹ thuật huấn luyện], [Mixed precision, etc.], [✓ Đầy đủ], [Đạt],
    [Tài nguyên], [GPU ≥ 8GB], [✓ Tesla T4 16GB], [Đạt]
  ),
  caption: "Đánh giá mức độ hoàn thành các yêu cầu đề tài"
)
== Kết quả chính đạt được

=== Hiệu suất mô hình

- Accuracy: 63.20% trên tập test, vượt 18% so với SVM và 4.8% so với mBERT
- Quá trình huấn luyện: Hội tụ ổn định trong 5 epochs, validation accuracy đạt 87.32%
- Khả năng phân loại: Hiệu suất cao nhất với cảm xúc Vui vẻ (F1: 72%), thấp nhất với Tức giận (F1: 44.74%)

== Hạn chế và nguyên nhân

=== Nguyên nhân chưa đạt yêu cầu hiệu suất

1. Mất cân bằng dữ liệu: Vui vẻ chiếm 33.4% trong khi Tức giận chỉ 17.8%
2. Kích thước dữ liệu hạn chế: 6.927 mẫu chưa đủ lớn cho deep learning
3. Độ phức tạp ngôn ngữ mạng xã hội: Teen code, từ viết tắt, cảm xúc hỗn hợp
4. Performance gap: Chênh lệch 24% giữa validation và test cho thấy overfitting

=== Thách thức kỹ thuật

- Khó khăn trong xử lý cảm xúc mỉa mai và gián tiếp
- Thiếu ngữ cảnh văn hóa trong mô hình pre-trained
- Hạn chế tài nguyên tính toán cho thử nghiệm mô hình lớn hơn

== Hướng phát triển

- Thu thập thêm dữ liệu cân bằng từ nhiều nguồn
- Áp dụng back-translation và paraphrasing
- Kết hợp nhiều mô hình để cải thiện hiệu suất