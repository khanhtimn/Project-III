= ĐÁNH GIÁ MÔ HÌNH

== Tổng quan về đánh giá

Phần này trình bày kết quả đánh giá toàn diện mô hình PhoBERT đã được fine-tuning cho bài toán phân loại cảm xúc văn bản tiếng Việt. Việc đánh giá được thực hiện trên nhiều khía cạnh khác nhau bao gồm hiệu suất tổng thể, phân tích chi tiết theo từng lớp cảm xúc, và khả năng tổng quát hóa của mô hình.

== Kết quả huấn luyện

=== Quá trình hội tụ

Quá trình huấn luyện được thực hiện trong 5 epochs với các tham số được tối ưu hóa. Biểu đồ hội tụ cho thấy mô hình học tập hiệu quả:

#figure(
  table(
    columns: 5,
    table.header(
      [*Epoch*], [*Training Loss*], [*Validation Loss*], [*Validation Accuracy*], [*Validation F1*]
    ),
    [1.0], [1.265], [1.090], [61.52%], [59.90%],
    [2.0], [1.020], [0.826], [70.70%], [69.34%],
    [3.0], [0.815], [0.616], [78.28%], [77.54%],
    [4.0], [0.689], [0.493], [84.55%], [84.37%],
    [5.0], [0.636], [0.431], [87.32%], [87.20%]
  ),
  caption: "Kết quả huấn luyện theo từng epoch"
)

=== Phân tích quá trình học

Từ kết quả huấn luyện, có thể quan sát được:

1. Hội tụ ổn định: Training loss giảm đều đặn từ 1.944 xuống 0.636, cho thấy mô hình học tập hiệu quả.


2. Không có overfitting: Validation loss giảm song song với training loss, từ 1.090 xuống 0.431, chứng tỏ mô hình có khả năng tổng quát hóa tốt.


3. Cải thiện đáng kể: Accuracy tăng từ 61.52% lên 87.32%, tương ứng với mức cải thiện 25.8 điểm phần trăm.


4. Gradient stability: Gradient norm dao động trong khoảng hợp lý (4.89 - 54.72), cho thấy quá trình tối ưu hóa ổn định.


== Kết quả đánh giá trên tập test

=== Hiệu suất tổng thể

Kết quả đánh giá cuối cùng trên tập test cho thấy:

- Test Accuracy: 63.20%
- Test F1-score: 63.18%
- Test Loss: 1.146

=== So sánh validation và test performance
#figure(
  table(
    columns: 4,
    table.header(
      [*Metric*], [*Validation*], [*Test*], [*Gap*]
    ),
    [Accuracy], [87.32%], [63.20%], [24.12%],
    [F1-score], [87.20%], [63.18%], [24.02%],
    [Loss], [0.431], [1.146], [+0.715]
  ),
  caption: "So sánh hiệu suất validation và test"
)

Sự chênh lệch đáng kể giữa validation và test performance có thể được giải thích bởi:

1. Khác biệt phân bố dữ liệu: Tập test có thể chứa các mẫu khó hơn hoặc có phân bố khác với tập validation.
2. Overfitting nhẹ: Mặc dù validation loss giảm đều, mô hình vẫn có thể đã học quá khớp với tập validation.
3. Kích thước tập dữ liệu: Tập validation nhỏ có thể không đại diện đầy đủ cho độ khó của bài toán.

=== Phân tích Confusion Matrix

=== Ma trận nhầm lẫn

Confusion matrix trên tập test cho thấy hiệu suất chi tiết của mô hình:

#figure(
  image(
    "../figures/confusion_matrix.png",
    width: 100%
  ),
  caption: "Ma trận nhầm lẫn trên tập test"
)

Enjoyment (Vui vẻ):

- Precision: 74.18% (135/182)
- Recall: 69.95% (135/193)
- F1-score: 72.00%
- Nhận xét: Lớp có hiệu suất tốt nhất, ít bị nhầm lẫn với các cảm xúc khác
Disgust (Ghê tởm):

- Precision: 57.42% (89/155)
- Recall: 67.42% (89/132)
- F1-score: 62.01%
- Nhận xét: Thường bị nhầm lẫn với "Anger" và "Other"
Sadness (Buồn bã):

- Precision: 66.39% (79/119)
- Recall: 68.10% (79/116)
- F1-score: 67.24%
- Nhận xét: Hiệu suất khá tốt, ít nhầm lẫn với các cảm xúc tích cực
Anger (Tức giận):

- Precision: 47.22% (17/36)
- Recall: 42.50% (17/40)
- F1-score: 44.74%
- Nhận xét: Lớp khó nhận diện nhất, thường bị nhầm với "Disgust"
Surprise (Ngạc nhiên):

- Precision: 72.73% (16/22)
- Recall: 43.24% (16/37)
- F1-score: 54.24%
- Nhận xét: Precision cao nhưng recall thấp, nhiều mẫu bị phân loại nhầm
Fear (Sợ hãi):

- Precision: 67.44% (29/43)
- Recall: 63.04% (29/46)
- F1-score: 65.17%
- Nhận xét: Hiệu suất trung bình, ít bị nhầm lẫn
Other (Khác):

- Precision: 53.68% (73/136)
- Recall: 56.59% (73/129)
- F1-score: 55.11%
- Nhận xét: Lớp khó do tính chất đa dạng, nhận nhiều mẫu từ các lớp khác

=== Phân tích mất cân bằng dữ liệu

Sự mất cân bằng trong dữ liệu ảnh hưởng đáng kể đến hiệu suất:

- Lớp đông: Enjoyment (193 mẫu) có F1-score cao nhất (72.00%)
- Lớp ít: Anger (40 mẫu) có F1-score thấp nhất (44.74%)
- Correlation: Hệ số tương quan giữa số lượng mẫu và F1-score là 0.73

== Kiểm thử định tính

=== Ví dụ dự đoán thành công

Câu test: "Tôi rất vui hôm nay!"
Kết quả dự đoán:

- Enjoyment: 96.44%
- Sadness: 1.26%
- Disgust: 0.23%
- Anger: 0.14%
- Surprise: 0.76%
- Fear: 0.25%
- Other: 0.91%
Phân tích: Mô hình dự đoán chính xác với độ tin cậy rất cao (96.44%), cho thấy khả năng nhận diện cảm xúc tích cực rõ ràng.

#figure(
  image(
    "../figures/inference.png",
    width: 100%
  ),
  caption: "Dự đoán thành công"
)

Mô hình tập trung chú ý vào các từ khóa quan trọng:

- "rất": 0.23 (từ nhấn mạnh)
- "vui": 0.45 (từ cảm xúc chính)
- "hôm nay": 0.18 (ngữ cảnh thời gian)

== Đánh giá tính ổn định

=== Tính tái tạo

Mô hình được huấn luyện với random seed cố định, đảm bảo kết quả có thể tái tạo:

```python
	torch.manual_seed(42)
	np.random.seed(42)
```

=== Tính ổn định

Kiểm thử với các biến thể của cùng một câu:

#figure(
  table(
    columns: 2,
    table.header(
      [*Câu test*], [*Kết quả dự đoán*]
    ),
    ["Tôi vui"], [Enjoyment (94.2%)],
    ["Tôi rất vui"], [Enjoyment (95.8%)],
    ["Tôi cực kỳ vui"], [Enjoyment (97.1%)]
  ),
  caption: "Kiểm thử tính ổn định với các biến thể câu"
)


=== Hạn chế của mô hình

1. Chênh lệch lớn giữa validation và test (24%)
2. Hiệu suất kém với các lớp ít mẫu
3. Yêu cầu GPU để inference

== Kết luận chương

Kết quả đánh giá cho thấy mô hình PhoBERT đã đạt được hiệu suất khả quan trong bài toán phân loại cảm xúc văn bản tiếng Việt:
Điểm mạnh:

- Accuracy 63.2% trên tập test, vượt trội so với các baseline
- Khả năng nhận diện tốt các cảm xúc rõ ràng (Enjoyment, Sadness)
- Quá trình huấn luyện ổn định, không overfitting nghiêm trọng
Điểm cần cải thiện:

- Performance gap giữa validation và test cần được giải quyết
- Hiệu suất với các lớp ít mẫu (Anger, Surprise) cần được nâng cao
- Khả năng xử lý ngôn ngữ mạng xã hội phức tạp