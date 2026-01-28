#import "/utils/todo.typ": TODO

=  TỔNG QUAN  ĐỀ  TÀI

== Giới thiệu bài toán
Phân loại cảm xúc là một bài toán quan trọng trong lĩnh vực Xử lý Ngôn ngữ Tự nhiên (Natural Language Processing - NLP), tập trung vào việc xác định và phân tích cảm xúc biểu đạt trong văn bản.

Phân loại cảm xúc văn bản (Text Emotion Classification) không chỉ đơn thuần là việc xác định tính tích cực hay tiêu cực của một đoạn văn, mà còn bao gồm việc nhận diện các trạng thái cảm xúc phức tạp và tinh tế như vui vẻ, buồn bã, tức giận, sợ hãi, ngạc nhiên và ghê tởm.

Sự bùng nổ của các nền tảng mạng xã hội và thương mại điện tử đã tạo ra một lượng dữ liệu văn bản khổng lồ chứa đựng thông tin cảm xúc phong phú từ người dùng. Theo thống kê, đến cuối năm 2019, Việt Nam đã có khoảng 48 triệu người dùng mạng xã hội, tạo ra một nguồn dữ liệu văn bản tiếng Việt đồ sộ cần được khai thác và phân tích. Việc hiểu được cảm xúc của người dùng không chỉ có giá trị trong nghiên cứu học thuật mà còn mang lại những ứng dụng thực tiễn quan trọng trong các lĩnh vực như:

- *Chăm sóc khách hàng*: Phân tích phản hồi, góp phần cải thiện chất lượng dịch vụ 

- *Mạng xã hội*: Theo dõi xu hướng cảm xúc của người dùng để phát hiện các vấn đề tiềm ẩn hoặc nắm bắt cơ hội truyền thông.

- *Tư vấn tâm lý*: Phát hiện và phân tích trạng thái cảm xúc, hỗ trợ trong quá trình chẩn đoán và trị liệu.

Tiếng Việt, với tư cách là một ngôn ngữ thuộc nhóm tài nguyên hạn chế (low-resource language), đặt ra những thách thức đặc biệt cho bài toán phân loại cảm xúc văn bản. Khác với các ngôn ngữ Indo-European, tiếng Việt có cấu trúc ngữ pháp phức tạp với đặc điểm đơn lập, trong đó cùng một từ có thể mang nhiều nghĩa khác nhau tùy thuộc vào ngữ cảnh và thanh điệu. Sự đa dạng về từ vựng, cách diễn đạt cảm xúc, và các biểu hiện ngôn ngữ địa phương tạo nên một không gian ngữ nghĩa phong phú nhưng cũng đầy thách thức.

Hơn nữa, việc biểu đạt cảm xúc trong tiếng Việt thường mang tính ngầm ẩn và phụ thuộc nhiều vào văn hóa, khiến cho các mô hình truyền thống dựa trên từ điển cảm xúc hoặc các phương pháp học máy cổ điển gặp khó khăn trong việc nắm bắt được những sắc thái cảm xúc tinh tế này.


== Mục tiêu của đề tài
Đề tài này nhằm mục đích khai thác sức mạnh của mô hình PhoBERT để giải quyết bài toán phân loại cảm xúc văn bản tiếng Việt một cách hiệu quả. Cụ thể, mục tiêu bao gồm:

- Áp dụng PhoBERT trong nhận diện cảm xúc văn bản tiếng Việt.

- Khảo sát và tối ưu các kỹ thuật fine-tuning để nâng cao hiệu suất mô hình.

- Đánh giá mô hình trên các tập dữ liệu chuẩn và so sánh với các phương pháp hiện có.

- Phân tích ưu nhược điểm của mô hình trong bối cảnh tiếng Việt.

== Phạm vi nghiên cứu của đề tài
=== Phạm vi dữ liệu và ngôn ngữ
Ngôn ngữ nghiên cứu: Văn bản tiếng Việt, với đặc điểm ngữ pháp đơn lập và biểu hiện cảm xúc phong phú, ngầm định.

Nguồn dữ liệu: Chủ yếu sử dụng tập dữ liệu UIT-VSMEC gồm 6.927 câu, gán nhãn theo 7 lớp cảm xúc (vui vẻ, buồn bã, tức giận, sợ hãi, ngạc nhiên, ghê tởm, và khác).

Bổ sung dữ liệu: Có thể mở rộng sang các tập khác như UIT-VSFC hoặc các corpus cảm xúc công khai khác.

=== Phạm vi về mô hình và phương pháp

- *Mô hình chính*: PhoBERT-base và PhoBERT-large, dựa trên kiến trúc RoBERTa, tối ưu hóa cho tiếng Việt.

- *Chiến lược fine-tuning*: Khảo sát các kỹ thuật như full fine-tuning, layer freezing, learning rate scheduling, gradient clipping, và regularization (dropout, label smoothing).

- *Thiết kế phân loại*: So sánh các cấu trúc classification head (linear, MLP) và hàm loss (cross-entropy, focal loss).

- *Tăng cường dữ liệu*: Ứng dụng các kỹ thuật như synonym replacement, back-translation để mở rộng tập huấn luyện.

=== Phạm vi đánh giá
- *Chỉ số đánh giá*: Sử dụng accuracy, precision, recall, macro-F1, weighted-F1 và confusion matrix.

- *Chiến lược đánh giá*: Áp dụng stratified k-fold cross-validation và hold-out test set; thực hiện nhiều lần huấn luyện với các random seeds khác nhau.

- *Phân tích kết quả*: Bao gồm error analysis, attention visualization và đánh giá khả năng tổng quát hóa.

== Yêu cầu của đề tài
=== Yêu cầu về hiệu suất mô hình
- *Accuracy*: tối thiểu 85% trên tập test; hướng tới 90%.

- *Macro-F1*: tối thiểu 75%, để đảm bảo hiệu quả với các lớp cảm xúc ít xuất hiện.

- *Weighted-F1*: tối thiểu 80%, phản ánh hiệu suất tổng thể có trọng số.

=== Yêu cầu về kiến trúc và triển khai
- Bắt buộc sử dụng PhoBERT làm mô hình chính.

- Áp dụng các kỹ thuật huấn luyện hiệu quả như mixed precision training, gradient accumulation, và early stopping.

- Huấn luyện hoàn chỉnh mô hình trên GPU đơn lẻ (≥ 8GB VRAM) trong thời gian hợp lý.

=== Yêu cầu về dữ liệu và tiền xử lý
- Chia tập dữ liệu theo tỷ lệ 80-10-10 (train-validation-test), đảm bảo phân bố đồng đều các nhãn.

- Áp dụng chuẩn hóa văn bản, xử lý dấu thanh, ký tự đặc biệt, từ viết tắt và biểu tượng cảm xúc.

- Sử dụng tokenizer PhoBERT với encoding BPE và độ dài chuỗi tối đa 512 tokens.

=== Yêu cầu về đánh giá và phân tích
- So sánh hiệu suất với các baseline như SVM, Random Forest (TF-IDF), CNN, BiLSTM, mBERT, và XLM-R.

- Phân tích lỗi theo lớp cảm xúc, độ dài câu, biểu hiện ngôn ngữ.

- Thực hiện attention visualization và statistical significance testing.

=== Yêu cầu về tái tạo và báo cáo
- Cung cấp toàn bộ mã nguồn, hyperparameters và hướng dẫn tái tạo.

- Thực hiện ablation study để đánh giá đóng góp của từng thành phần trong pipeline.

- Báo cáo đầy đủ về tài nguyên tính toán sử dụng, hiệu suất huấn luyện và các giới hạn thực tiễn.

== Phương pháp nghiên cứu
Quy trình nghiên cứu gồm ba giai đoạn chính:

- *Chuẩn bị dữ liệu*: Tiền xử lý văn bản tiếng Việt, xử lý mất cân bằng dữ liệu, và áp dụng các kỹ thuật tăng cường dữ liệu như synonym replacement, back-translation.

- *Huấn luyện mô hình*: Sử dụng PhoBERT với kiến trúc classification head phù hợp, kết hợp kỹ thuật regularization, tuning siêu tham số và checkpointing.

- *Đánh giá và phân tích*: Sử dụng các chỉ số định lượng (accuracy, F1-score), cùng các phương pháp định tính như error analysis và attention visualization để phân tích kết quả.

== Kết luận chương
Chương này đã trình bày tổng quan về bối cảnh, mục tiêu, phạm vi, yêu cầu và phương pháp nghiên cứu của đề tài. Việc kết hợp mô hình PhoBERT với đặc thù tiếng Việt hứa hẹn đem lại hiệu quả cao trong phân loại cảm xúc, đồng thời đóng góp vào việc phát triển các ứng dụng AI ngôn ngữ cho tiếng Việt. Những nội dung này sẽ là nền tảng cho các chương tiếp theo của báo cáo.

