= Kiến trúc mô hình 

== Tổng quan về Transformer

Transformer là kiến trúc mạng neural được giới thiệu trong bài báo "Attention Is All You Need" @vaswani2023, đánh dấu một bước ngoặt quan trọng trong lĩnh vực xử lý ngôn ngữ tự nhiên. Khác với các kiến trúc truyền thống như RNN hay LSTM xử lý tuần tự, Transformer sử dụng cơ chế tự chú ý để xử lý toàn bộ chuỗi đầu vào song song, cho phép mô hình nắm bắt được các mối quan hệ xa trong văn bản một cách hiệu quả.
Kiến trúc Transformer gồm hai thành phần chính:

- Encoder: Mã hóa chuỗi đầu vào thành các biểu diễn ngữ cảnh
- Decoder: Sinh ra chuỗi đầu ra dựa trên biểu diễn từ encoder
Đối với bài toán phân loại cảm xúc, chúng ta chỉ sử dụng phần Encoder của Transformer, tương tự như trong BERT.

== Cơ chế Self-Attention

Self-attention là cốt lõi của kiến trúc Transformer, cho phép mỗi vị trí trong chuỗi "chú ý" đến tất cả các vị trí khác để tạo ra biểu diễn ngữ cảnh phong phú.

=== Công thức toán học

Cơ chế attention được định nghĩa như sau:
$
"Attention"(Q, K, V) = "softmax"(frac(Q K^T, sqrt(d_k))) V
$

Trong đó:

- $Q$ (Query): Ma trận truy vấn có kích thước $[n times d_k]$
- $K$ (Key): Ma trận khóa có kích thước $[n times d_k]$
- $V$ (Value): Ma trận giá trị có kích thước $[n times d_v]$
- $d_k$: Chiều của vector key/query
- $n$: Độ dài chuỗi đầu vào

=== Quá trình tính toán Self-Attention

1. *Tạo ma trận Q, K, V*: Từ đầu vào $X$, tạo ra ba ma trận thông qua phép nhân với các ma trận trọng số học được:
   $
   Q = X W^Q, quad K = X W^K, quad V = X W^V
   $

2. *Tính điểm attention*: Tính độ tương quan giữa mỗi cặp vị trí:
   $
   "scores" = frac(Q K^T, sqrt(d_k))
   $

3. *Chuẩn hóa*: Áp dụng softmax để chuyển điểm thành xác suất:
   $
   "weights" = "softmax"("scores")
   $

4. *Tổng có trọng số*: Tính tổng có trọng số của các giá trị:
   $
   "output" = "weights" dot V
   $

=== Ý nghĩa của Self-Attention trong phân loại cảm xúc

Đối với bài toán phân loại cảm xúc văn bản tiếng Việt, self-attention mang lại những lợi ích quan trọng:

- Nắm bắt mối quan hệ xa: Có thể liên kết các từ cảm xúc với đối tượng được đề cập ở xa trong câu
- Xử lý ngữ pháp phức tạp: Hiểu được cấu trúc đảo ngữ, câu ghép phức tạp trong tiếng Việt
- Phân biệt ngữ cảnh: Cùng một từ có thể mang ý nghĩa khác nhau tùy ngữ cảnh

== Multi-Head Attention

Multi-Head Attention mở rộng cơ chế self-attention bằng cách sử dụng nhiều "đầu chú ý" song song, cho phép mô hình tập trung vào các khía cạnh khác nhau của thông tin.

$
"MultiHead"(Q, K, V) = "Concat"("head"_1, ..., "head"_h)W^O
$
Trong đó mỗi head được tính:
$
"head"_i = "Attention"(Q W_i^Q, K W_i^K, V W_i^V)
$
Với:

- $h$: Số lượng attention heads
- $W_i^Q, W_i^K, W_i^V$: Ma trận trọng số cho head thứ $i$
- $W^O$: Ma trận trọng số đầu ra

*Lợi ích của Multi-Head Attention*

- Đa dạng hóa thông tin: Mỗi head có thể học các pattern khác nhau
- Tăng khả năng biểu diễn: Kết hợp nhiều góc nhìn về cùng một thông tin
- Ổn định huấn luyện: Giảm thiểu rủi ro của việc chỉ dựa vào một cơ chế attention

== Position Encoding

Vì Transformer không có cơ chế xử lý tuần tự như RNN, cần có cách để mô hình hiểu được vị trí của các từ trong câu.

=== Sinusoidal Position Encoding

Transformer sử dụng hàm sin và cos để mã hóa vị trí:
$
P E_("pos", 2i) = sin("pos" / 10000^(2i / d_"model"))
$
$
P E_("pos", 2i+1) = cos("pos" / 10000^(2i / d_"model"))
$
Trong đó:

- $"pos"$: Vị trí của từ trong câu
- $i$: Chiều của vector embedding
- $d_"model"$: Kích thước của model

== Feed-Forward Networks

Mỗi lớp trong Transformer encoder chứa một mạng feed-forward đơn giản:
$
"FFN"(x) = max(0, x W_1 + b_1) W_2 + b_2
$
Mạng này có vai trò:

- Biến đổi phi tuyến: Tăng khả năng biểu diễn của mô hình
- Tăng chiều: Thường mở rộng từ $d_"model"$ lên $4 times d_"model"$ rồi thu nhỏ lại
- Xử lý từng vị trí: Áp dụng cùng một phép biến đổi cho mọi vị trí

== Kiến trúc Encoder hoàn chỉnh

Một lớp Transformer Encoder hoàn chỉnh bao gồm:

1. Multi-Head Self-Attention
2. Feed-Forward Network
3. Residual connection
4. Layer normalization

Công thức tổng quát cho một encoder layer:
$
"output"_1 &= "LayerNorm"(x + "MultiHeadAttention"(x)) \
"output"_2 &= "LayerNorm"("output"_1 + "FFN"("output"_1))
$

== Mô hình BERT

=== Tổng quan về BERT

BERT dựa trên kiến trúc Transformer Encoder, loại bỏ phần Decoder vì mục tiêu chính là tạo ra các biểu diễn ngữ cảnh tốt cho các tác vụ downstream như phân loại văn bản, nhận diện thực thể có tên, và trả lời câu hỏi.

=== Kiến trúc BERT

==== Cấu trúc tổng thể

BERT sử dụng kiến trúc Transformer Encoder với các thông số kỹ thuật sau:

#figure(
  table(
    columns: 3,
    table.header(
      [*Thông số*], [*BERT-Base*], [*BERT-Large*]
    ),
    [Số lớp Transformer ($L$)], [12], [24],
    [Kích thước hidden state ($H$)], [768], [1024],
    [Số attention heads ($A$)], [12], [16],
    [Tổng số tham số], [110M], [340M]
  ),
  caption: "So sánh thông số kỹ thuật BERT-Base và BERT-Large"
)

==== Biểu diễn đầu vào

BERT sử dụng ba loại embedding được cộng lại để tạo biểu diễn đầu vào:

$
"Input" = "Token Embeddings" + "Segment Embeddings" + "Position Embeddings"
$

*Token Embeddings:*

- Mỗi token được ánh xạ thành vector $d_"model"$ chiều

*Segment Embeddings:*

- Phân biệt các câu khác nhau trong cùng một input
- Segment A (câu đầu): embedding vector $E_A$
- Segment B (câu thứ hai): embedding vector $E_B$

*Position Embeddings:*

- Khác với Transformer gốc sử dụng sinusoidal encoding
- BERT học position embeddings như các tham số của mô hình
- Hỗ trợ tối đa 512 positions

=== Special Tokens

BERT sử dụng các token đặc biệt:

- [CLS]: Classification token, luôn ở đầu mỗi sequence
- [SEP]: Separator token, phân tách các câu
- [MASK]: Mask token, thay thế các token bị che trong pre-training
- [PAD]: Padding token, đệm các sequence ngắn
- [UNK]: Unknown token, thay thế các token không có trong vocabulary

== Mục tiêu tiền huấn luyện

BERT được tiền huấn luyện với hai mục tiêu chính, cho phép mô hình học được biểu diễn ngữ cảnh hai chiều:

=== Masked Language Model (MLM)

MLM là kỹ thuật cốt lõi giúp BERT học được biểu diễn hai chiều. Quá trình này bao gồm:
Định dạng đầu vào:

- Chọn ngẫu nhiên 15% tokens trong chuỗi đầu vào
- Trong số 15% tokens được chọn:
	- 80% được thay bằng [MASK]
	- 10% được thay bằng token ngẫu nhiên khác
	- 10% giữ nguyên token gốc
Hàm mục tiêu:
$
cal(L)_"MLM" = -sum_(i in cal(M)) log P(x_i | x_(backslash cal(M)))
$
Trong đó:

- $cal(M)$: Tập hợp các vị trí bị mask
- $x_(backslash cal(M))$: Sequence với các token bị mask
- $P(x_i | x_(backslash cal(M)))$: Xác suất dự đoán token $x_i$ dựa trên ngữ cảnh

=== Next Sentence Prediction (NSP)

NSP giúp BERT hiểu được mối quan hệ giữa các câu:
Định dạng đầu vào:

	[CLS] Sentence A [SEP] Sentence B [SEP]

Objective:

- 50% trường hợp: Sentence B thực sự theo sau Sentence A (IsNext)
- 50% trường hợp: Sentence B là câu ngẫu nhiên (NotNext)
Hàm Loss:
$
cal(L)_"NSP" = -log P("IsNext" | ["CLS"])
$

== Hiểu ngữ cảnh hai chiều

Mô hình truyền thống chỉ có thể nhìn thấy ngữ cảnh từ trái sang phải:

$h_i = "Transformer"(h_(i-1)), quad i in [1, n]$

BERT có thể nhìn thấy ngữ cảnh từ cả hai hướng:

$h_i = "Transformer"(h_1, h_2, ..., h_n), quad forall i$

== Kiến trúc fine-tuning

=== Bài toán phân loại cảm xúc

Đối với bài toán phân loại cảm xúc, BERT sử dụng [CLS] token:
$
"logits" = W dot h_("CLS") + b
$
$
P("class" | "input") = "softmax"("logits")
$
Trong đó:

- $h_("CLS")$: Hidden state của [CLS] token từ layer cuối
- $W in RR^(K times H)$: Ma trận trọng số classification head
- $K$: Số lượng classes (7 cho bài toán cảm xúc)
- $H$: Kích thước hidden state (768 cho BERT-Base)

=== Hàm Loss cho bài toán phân loại

$
cal(L)_"classification" = -sum_(i=1)^N sum_(k=1)^K y_(i,k) log P(k | x_i)
$

Trong đó:

- $N$: Số lượng mẫu
- $K$: Số lượng classes
- $y_(i,k)$: One-hot encoding của label thực tế

== Attention Patterns trong BERT

Các attention heads trong BERT học được các pattern khác nhau @devlin2019bertpretrainingdeepbidirectional:
\ *Syntactic Heads:*

- Tập trung vào mối quan hệ ngữ pháp (chủ-vị, định-trung tâm)
- Quan trọng cho việc hiểu cấu trúc câu tiếng Việt

*Semantic Heads:*

- Liên kết các từ có nghĩa tương quan
- Hữu ích cho việc nhận diện cảm xúc

*Positional Heads:*

- Chú ý đến vị trí tương đối của các từ
- Quan trọng với trật tự từ cố định trong tiếng Việt


Có thể trực quan hóa attention weights để hiểu cách BERT xử lý cảm xúc:
$
"Attention"_(i,j) = frac(exp(e_(i,j)), sum_(k=1)^n exp(e_(i,k)))
$
Trong đó $e_(i,j)$ là attention score giữa token $i$ và token $j$.

== Mặt nạ chú ý (Attention Mask)
Mặt nạ chú ý là một tensor nhị phân (0 và 1) có cùng kích thước với chuỗi đầu vào, cho biết mô hình nên "chú ý" (attend) vào token nào và bỏ qua token nào trong quá trình tính toán tự chú ý (self-attention).

Giá trị mặt nạ chú ý:

- 1: Token thực sự (mô hình cần xử lý)
- 0: Token đệm (padding) (mô hình bỏ qua)

Tại sao cần mặt nạ chú ý?

Khi xử lý lô dữ liệu, các câu có độ dài khác nhau cần được đệm về cùng một độ dài. Nếu không có mặt nạ chú ý, mô hình sẽ "chú ý" cả vào các token đệm, gây ra:

- Nhiễu trong quá trình học
- Giảm chất lượng biểu diễn
- Dự đoán không chính xác
#figure(
  table(
    columns: 3,
    table.header(
      [*Câu gốc*], [*Token hóa + Đệm*], [*Mặt nạ chú ý*]
    ),
    ["Sản phẩm rất tốt"], [[CLS] Sản phẩm rất tốt [SEP] [PAD] [PAD]], [[1, 1, 1, 1, 1, 0, 0]],
    ["Tốt"], [[CLS] Tốt [SEP] [PAD] [PAD] [PAD] [PAD]], [[1, 1, 1, 0, 0, 0, 0]]
  ),
  caption: "Ví dụ token hóa, đệm và mặt nạ chú ý"
)
== Mô hình PhoBERT

=== Tổng quan về PhoBERT

PhoBERT là mô hình BERT đầu tiên được thiết kế và tối ưu hóa đặc biệt cho tiếng Việt @nguyen2020phobertpretrainedlanguagemodels. PhoBERT kế thừa toàn bộ kiến trúc và phương pháp huấn luyện của BERT gốc, nhưng được điều chỉnh để phù hợp với đặc thù ngôn ngữ học và văn hóa của tiếng Việt.

Sự ra đời của PhoBERT đánh dấu một bước tiến quan trọng trong việc phát triển các công nghệ xử lý ngôn ngữ tự nhiên cho tiếng Việt, giải quyết những hạn chế của các mô hình đa ngôn ngữ như mBERT trong việc xử lý các đặc thù riêng biệt của tiếng Việt. Cụ thể, các mô hình đa ngôn ngữ trước đây như Multilingual BERT (mBERT) được huấn luyện trên 104 ngôn ngữ, bao gồm tiếng Việt, nhưng gặp phải những hạn chế sau:

- Khả năng biểu diễn cho từng ngôn ngữ bị pha loãng do phải chia sẻ tham số cho nhiều ngôn ngữ
- Chỉ có khoảng 1% từ vựng dành cho tiếng Việt
- Thiếu hiểu biết sâu về văn hóa và ngữ cảnh Việt Nam
- Dữ liệu huấn luyện chủ yếu từ Wikipedia, không phản ánh đầy đủ ngôn ngữ, văn hóa, và ngữ cảnh Việt Nam

== Kiến trúc PhoBERT

PhoBERT được phát hành với hai phiên bản:

#figure(
  table(
    columns: 3,
    table.header(
      [*Thông số*], [*PhoBERT-base*], [*PhoBERT-large*]
    ),
    [Số lớp Transformer ($L$)], [12], [24],
    [Kích thước hidden state ($H$)], [768], [1024],
    [Số attention heads ($A$)], [12], [16],
    [Kích thước intermediate], [3072], [4096],
    [Tổng số tham số], [135M], [370M],
    [Kích thước vocabulary], [64,001 tokens], [64,001 tokens]
  ),
  caption: "So sánh thông số kỹ thuật PhoBERT-Base và PhoBERT-Large"
)

=== Đặc điểm kỹ thuật của PhoBERT

==== Xử lý thanh điệu tiếng Việt

PhoBERT được thiết kế để xử lý hiệu quả hệ thống thanh điệu phức tạp của tiếng Việt:
Thanh điệu và nghĩa từ:

	ma (ma quỷ) - mà (nhưng) - má (mẹ) - mả (mộ) - mã (số hiệu) - mạ (lúa)

Cơ chế xử lý:

- BPE học được các pattern thanh điệu phổ biến
- Attention mechanism nắm bắt được mối quan hệ giữa các biến thể thanh điệu
- Context embedding giúp phân biệt nghĩa dựa trên ngữ cảnh

==== Xử lý từ ghép và cụm từ

Từ ghép tiếng Việt:

#figure(
  table(
    columns: 2,
    table.header(
      [*Từ ghép*], [*Segmentation*]
    ),
    [máy tính], [`["máy_", "tính"]`],
    [học sinh], [`["học_", "sinh"]`],
    [trường đại học], [`["trường_", "đại_", "học"]`]
  ),
  caption: "Ví dụ về segmentation từ ghép tiếng Việt trong PhoBERT"
)

Lợi ích:

- Giữ được ý nghĩa ngữ nghĩa của từ ghép
- Giảm thiểu tình trạng over-segmentation
- Tăng hiệu quả trong việc biểu diễn khái niệm phức tạp

==== Xử lý ngôn ngữ mạng xã hội

Teen code và từ viết tắt:

	#figure(
	  table(
	    columns: 2,
	    table.header(
	      [*Teen code*], [*Từ gốc*]
	    ),
	    ["ko"], ["không"],
	    ["dc"], ["được"],
	    ["vs"], ["với"],
	    ["tks"], ["thanks"],
	    ["iu"], ["yêu"]
	  ),
	  caption: "Ví dụ teen code và từ viết tắt trong tiếng Việt"
	)

Emoji và emoticons:

- PhoBERT học được mối quan hệ giữa emoji và cảm xúc
- Xử lý được các biểu tượng cảm xúc phổ biến trong văn hóa Việt
- Tích hợp thông tin từ emoji vào phân tích cảm xúc

=== Token hóa trong PhoBERT
==== Quy trình BPE trong PhoBERT

*Dữ liệu huấn luyện BPE:*

- 20GB văn bản tiếng Việt (Wikipedia + VnExpress + Zing News)
- 145 triệu câu đã được phân đoạn từ
- 3 tỷ word tokens

*Đặc điểm quan trọng của PhoBERT BPE:*

- Xử lý dấu thanh: BPE học được các pattern dấu thanh phổ biến
  - Ví dụ: "à", "á", "ả", "ã", "ạ" được học như các subword riêng biệt

- Xử lý từ ghép: Từ đã phân đoạn được BPE xử lý hiệu quả hơn
  - "đại_học" → có thể được tokenize thành `["đại_", "học"]` hoặc `["đại_học"]`

- Marker kết thúc từ: Sử dụng `</w>` để phân biệt subword cuối từ
  - "học_sinh" → `["học_", "sinh</w>"]`

==== Cấu trúc vốn từ vựng PhoBERT

PhoBERT sử dụng từ vựng có kích thước 64,001 tokens, được xây dựng từ 20GB văn bản tiếng Việt đã phân đoạn từ. Từ vựng này được tổ chức theo cấu trúc phân tầng:

#figure(
  table(
    columns: (1fr, auto),
    align: (left, right),
    stroke: 0.5pt,
    [*Loại Token*], [*Số lượng*],
    [Token đặc biệt (Special Tokens)], [6 tokens],
    [Ký tự đơn (Single Characters)], [~300 tokens],
    [Từ con tiếng Việt phổ biến], [~15,000 tokens],
    [Thuật ngữ chuyên ngành], [~10,000 tokens],
    [Từ con hiếm], [~20,000 tokens],
    [Số và dấu câu], [~18,695 tokens],
    table.cell(colspan: 2, align: center)[*Tổng: 64,001 tokens*]
  ),
  caption: "Cấu trúc từ vựng PhoBERT"
)

===== Token đặc biệt trong PhoBERT

PhoBERT sử dụng 6 token đặc biệt chính để xử lý các tác vụ khác nhau:

1. *[CLS] Token (ID: 2)*
   - Mục đích: Token phân loại, chứa thông tin tổng hợp của toàn câu
   - Vị trí: Luôn ở đầu mỗi chuỗi
   - Sử dụng: Dùng cho các tác vụ phân loại

   #align(left)[
   *Ví dụ: Phân loại cảm xúc*
   ```
   text = "Sản phẩm này rất tốt"
   Sau token hóa: [CLS] Sản phẩm này rất tốt [SEP]
   Vector của [CLS] sẽ được dùng để dự đoán cảm xúc: "tích cực"
   ```
   ]

2. *[SEP] Token (ID: 3)*
   - Mục đích: Phân tách các câu hoặc kết thúc chuỗi
   - Vị trí: Cuối mỗi câu
   - Sử dụng: Đánh dấu ranh giới câu

   #align(left)[
   *Ví dụ 1: Câu đơn*
   ```
   text = "Tôi thích món này"
   Kết quả: [CLS] Tôi thích món này [SEP]
   ```
   
   *Ví dụ 2: Cặp câu (cho tác vụ so sánh)*
   ```
   text1 = "Sản phẩm chất lượng"
   text2 = "Giá cả hợp lý"
   Kết quả: [CLS] Sản phẩm chất lượng [SEP] Giá cả hợp lý [SEP]
   ```
   ]

3. *[PAD] Token (ID: 0)*
   - Mục đích: Padding để chuẩn hóa độ dài chuỗi
   - Vị trí: Cuối chuỗi ngắn
   - Sử dụng: Đảm bảo tất cả chuỗi có cùng độ dài

   #align(left)[
   *Ví dụ với max_length = 10*
   ```
   text1 = "Tốt"
   text2 = "Rất hài lòng"
   
   Sau token hóa:
   text1: [CLS] Tốt [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
   text2: [CLS] Rất hài lòng [SEP] [PAD] [PAD] [PAD] [PAD] [PAD]
   ```
   ]

4. *[MASK] Token (ID: 4)*
   - Mục đích: Che giấu từ trong quá trình pre-training
   - Sử dụng: Chủ yếu trong giai đoạn huấn luyện mô hình
   - Fine-tuning: Ít sử dụng trực tiếp

   #align(left)[
   *Ví dụ trong pre-training*
   ```
   original = "Tôi rất thích sản phẩm này"
   masked   = "Tôi rất [MASK] sản phẩm này"
   ```
   Mô hình học dự đoán từ "thích" từ ngữ cảnh xung quanh
   ]

5. *[UNK] Token (ID: 1)*
   - Mục đích: Thay thế từ không có trong vốn từ vựng
   - Tần suất: Rất hiếm trong PhoBERT (~0.1%)
   - Lý do: BPE thường chia từ lạ thành từ con

   #align(left)[
   *Ví dụ với ký tự đặc biệt*
   ```
   text = "Tôi thích ∑∫∂∇ này"
   Có thể thành: [CLS] Tôi thích [UNK] này [SEP]
   Nhưng thường BPE sẽ chia:
   [CLS] Tôi thích ∑ ∫ ∂ ∇ này [SEP]
   ```
   ]

6. *Sentence Boundary: <s>, </s> (ID: 5, 6)*
   - Mục đích: Đánh dấu ranh giới câu (ít dùng)
   - Nguồn gốc: Kế thừa từ RoBERTa
   - Sử dụng: Thường được thay thế bởi [CLS] và [SEP]

== Fine-tuning và Transfer Learning

=== Khái niệm Fine-tuning

=== Định nghĩa và nguyên lý

Fine-tuning là quá trình điều chỉnh một mô hình đã được pre-train trên tập dữ liệu lớn để thích ứng với một tác vụ cụ thể (downstream task) trên tập dữ liệu nhỏ hơn. Trong ngữ cảnh của PhoBERT, fine-tuning là việc sử dụng các trọng số đã được học từ quá trình pre-training trên corpus tiếng Việt lớn để giải quyết bài toán phân loại cảm xúc cụ thể.
Nguyên lý hoạt động:
$
theta_"fine-tuned" = theta_"pre-trained" + Delta theta
$

Trong đó:

- $theta_"pre-trained"$: Tham số từ mô hình đã pre-train
- $Delta theta$: Sự thay đổi tham số trong quá trình fine-tuning
- $theta_"fine-tuned"$: Tham số cuối cùng cho tác vụ cụ thể

=== Transfer Learning trong NLP

Transfer Learning trong NLP dựa trên giả định rằng các biểu diễn ngôn ngữ học được từ tác vụ này có thể được chuyển giao và sử dụng cho tác vụ khác. Quá trình này bao gồm hai giai đoạn chính:
\ Giai đoạn 1: Pre-training (Không giám sát)

- Mô hình học các biểu diễn ngôn ngữ tổng quát
- Sử dụng tập dữ liệu lớn không có nhãn
- Mục tiêu: MLM và NSP cho BERT
Giai đoạn 2: Fine-tuning (Có giám sát)

- Điều chỉnh mô hình cho tác vụ cụ thể
- Sử dụng tập dữ liệu có nhãn nhỏ hơn
- Mục tiêu: Hàm loss tác vụ cụ thể

=== Lợi ích của Fine-tuning

Fine-tuning mang lại những ưu điểm vượt trội trong việc phát triển các mô hình xử lý ngôn ngữ tự nhiên, đặc biệt quan trọng đối với các ngôn ngữ có tài nguyên hạn chế như tiếng Việt. Về mặt hiệu quả tính toán, fine-tuning giúp giảm đáng kể thời gian huấn luyện từ hàng tuần hoặc tháng xuống chỉ còn vài giờ hoặc ngày, đồng thời yêu cầu tài nguyên tính toán thấp hơn nhiều so với việc huấn luyện từ đầu. Điều này cho phép các nhà nghiên cứu và tổ chức có ngân sách hạn chế có thể thực hiện được trên một GPU đơn lẻ.

Về hiệu suất mô hình, fine-tuning cho phép đạt được performance cao ngay cả khi chỉ có ít dữ liệu được gán nhãn, nhờ vào việc tận dụng hiệu quả kiến thức đã được học từ corpus lớn trong giai đoạn pre-training. Cơ chế này cũng giúp giảm thiểu hiện tượng overfitting thường xảy ra khi huấn luyện trên các tập dữ liệu nhỏ.

Khả năng ứng dụng linh hoạt của fine-tuning thể hiện qua việc dễ dàng thích ứng với các lĩnh vực khác nhau mà không cần kiến thức chuyên sâu về thiết kế kiến trúc mạng. Đồng thời, phương pháp này có thể được tùy chỉnh theo các yêu cầu cụ thể của từng bài toán, từ phân loại cảm xúc đến nhận diện thực thể có tên hay trả lời câu hỏi.

== Chiến lược Fine-tuning

=== Full Fine-tuning

Full fine-tuning cập nhật toàn bộ tham số của mô hình pre-trained:
$
cal(L)_"total" = cal(L)_"task" + lambda cal(L)_"reg"
$

Trong đó:

- $cal(L)_"task"$: Loss function của tác vụ cụ thể
- $cal(L)_"reg"$: Regularization term
- $lambda$: Regularization coefficient

=== Gradual Unfreezing

Gradual Unfreezing dựa trên nguyên lý rằng các lớp khác nhau trong mạng neural học các đặc trưng ở mức độ trừu tượng khác nhau. Cụ thể, các lớp sâu hơn (gần đầu ra) học các đặc trưng cấp cao và cụ thể cho tác vụ, trong khi các lớp nông hơn (gần đầu vào) học các đặc trưng tổng quát và cơ bản.

Chiến lược này bắt đầu bằng việc đóng băng (freeze) toàn bộ các lớp pre-trained và chỉ huấn luyện classifier head. Sau đó, theo thời gian huấn luyện, các lớp được giải phóng dần dần từ trên xuống (từ đầu ra về đầu vào). Quá trình này được điều khiển bởi hàm:

$
L_"unfreeze"(t) = floor(frac(t, T) times L_"total")
$

Trong đó:
- $L_"unfreeze"(t)$: Số lớp được giải phóng tại thời điểm $t$
- $T$: Tổng thời gian huấn luyện  
- $L_"total"$: Tổng số lớp trong mô hình

=== Layer-wise Learning Rate Decay

Áp dụng learning rate khác nhau cho các layer khác nhau:

$
l r_i = l r_"base" times alpha^(L-i)
$

Trong đó:

- $l r_i$: Learning rate cho layer $i$
- $l r_"base"$: Base learning rate
- $alpha$: Decay factor (thường 0.9-0.95)
- $L$: Tổng số layers
- $i$: Index của layer (0 = bottom layer)

== Tham số Fine-tuning

=== Learning Rate (Tốc độ học)

Learning rate đóng vai trò then chốt trong quá trình fine-tuning, quyết định mức độ điều chỉnh trọng số của mô hình trong mỗi bước cập nhật gradient. Về mặt toán học, learning rate kiểm soát mức độ của việc cập nhật tham số theo công thức:
$
theta_(t+1) = theta_t - eta nabla_theta cal(L)(theta_t)
$

Trong đó $theta_t$ là tham số tại bước $t$, $nabla_theta cal(L)(theta_t)$ là gradient của hàm loss tại điểm hiện tại. Đối với các mô hình pre-trained như PhoBERT, việc lựa chọn learning rate phù hợp đặc biệt quan trọng vì cần cân bằng giữa việc bảo tồn kiến thức đã học từ pre-training và khả năng thích ứng với tác vụ mới.

Khi learning rate quá cao, $Delta theta$ lớn có thể dẫn đến việc mất mát kiến thức pre-trained. Các nghiên cứu thực nghiệm cho thấy learning rate trong khoảng 1e-5 đến 5e-5 thường mang lại kết quả tối ưu cho các lớp BERT, trong khi classification head có thể sử dụng learning rate cao hơn từ 1e-4 đến 1e-3. 

Nguyên lý này dựa trên thực tế rằng các trọng số pre-trained đã gần với lời giải tối ưu cho các tác vụ ngôn ngữ tổng quát, được biểu diễn qua:

$
cal(L)_"downstream"(theta) &approx cal(L)_"downstream"(theta_"pre-trained") \
&+ nabla cal(L)^T (theta - theta_"pre-trained") \
&+ 1/2 (theta - theta_"pre-trained")^T H (theta - theta_"pre-trained")
$

Trong đó $H$ là ma trận Hessian, cho thấy trọng số pre-trained nằm gần điểm cực tiểu địa phương của tác vụ downstream.

=== Batch Size và Gradient Accumulation

Batch size ảnh hưởng trực tiếp đến tính ổn định, tốc độ hội tụ và khả năng tổng quát hóa của mô hình thông qua chất lượng ước lượng gradient. Gradient thực tế được ước lượng từ mini-batch:

$
hat(nabla)cal(L) = 1/B sum_(i=1)^B nabla cal(L)(x_i, y_i; theta)
$

Trong đó $B$ là batch size. Variance của gradient estimator tỷ lệ nghịch với batch size:

$
"Var"[hat(nabla)cal(L)] = sigma^2/B
$

Batch size nhỏ (8-16) mang lại một số lợi ích đáng kể: yêu cầu bộ nhớ thấp hơn.

Gradient accumulation cho phép mô phỏng batch size lớn với bộ nhớ hạn chế thông qua công thức:

$
hat(nabla)cal(L)_"accumulated" = 1/K sum_(k=1)^K hat(nabla)cal(L)_k
$

Trong đó $K$ là số accumulation steps, tạo ra effective batch size $B_"eff" = B times K$.

=== Số epochs và Early Stopping

Số epochs kiểm soát tổng lượng thông tin mà mô hình được expose, được định lượng qua tổng số lần cập nhật gradient:

$
N_"updates" = (N_"samples" times N_"epochs")/B_"eff"
$

Rủi ro overfitting tăng theo số epochs, được mô tả qua sự khác biệt giữa loss trên tập huấn luyện và loss trên tập kiểm tra:

$
"Gap" = cal(L)_"validation" - cal(L)_"training"
$

Early stopping sử dụng cơ chế độ kiên nhẫn (patience) để kết thúc huấn luyện khi hiệu suất kiểm tra không cải thiện:

$
"Stop if: " max_(t' in [t-p, t]) "Score"(t') - "Score"(t) < delta
$

Trong đó $p$ là độ kiên nhẫn, $delta$ là ngưỡng cải thiện tối thiểu.

== Kết luận chương

Chương này đã trình bày về thiết kế và triển khai mô hình PhoBERT cho bài toán phân loại cảm xúc văn bản tiếng Việt, từ nền tảng lý thuyết đến các kỹ thuật thực thi cụ thể.