= Triển khai mô hình

== Tổng quan về triển khai

Phần này trình bày chi tiết quá trình triển khai mô hình PhoBERT cho bài toán phân loại cảm xúc văn bản tiếng Việt, từ việc chuẩn bị môi trường phát triển đến việc huấn luyện và đánh giá mô hình.

== Môi trường phát triển và cấu hình hệ thống

=== Cấu hình phần cứng

Việc triển khai mô hình PhoBERT đòi hỏi tài nguyên tính toán đáng kể do kích thước lớn của mô hình và độ phức tạp của quá trình fine-tuning. Cấu hình hệ thống được sử dụng bao gồm:

- GPU: NVIDIA Tesla T4 với 16GB VRAM
- RAM: 32GB

=== Môi trường phần mềm

Môi trường phát triển được thiết lập trên Google Colab để tận dụng tài nguyên GPU. Các thư viện chính được sử dụng:

	#figure(
	  table(
	    columns: 2,
	    table.header(
	      [*Loại thư viện*], [*Thư viện*]
	    ),
	    [Deep Learning], [torch, transformers, datasets, evaluate],
	    [NLP Processing], [pyvi, gensim],
	    [Data Analysis], [pandas, numpy, scikit-learn],
	    [Visualization], [matplotlib, seaborn]
	  ),
	  caption: "Thư viện được sử dụng trong triển khai"
	)

== Chuẩn bị dữ liệu

=== Khám phá dữ liệu

Dữ liệu được tải từ bộ dữ liệu UIT-VSMEC đã được chia sẵn thành các tập train, validation và test.

```python
	def get_data(path):
	    df = pd.read_excel(path, sheet_name=None)['Sheet1']
	    df.columns = ['index', 'Emotion', 'Sentence']
	    df.drop(columns=['index'], inplace=True)
	    return df
	
	train_df = get_data('/content/drive/MyDrive/Project-2/data/train_nor_811.xlsx')
	valid_df = get_data('/content/drive/MyDrive/Project-2/data/valid_nor_811.xlsx')
	test_df = get_data('/content/drive/MyDrive/Project-2/data/test_nor_811.xlsx')
```

=== Phân tích phân bố dữ liệu

Phân tích phân bố nhãn cảm xúc trong tập huấn luyện cho thấy sự mất cân bằng đáng kể giữa các lớp:

#figure(
  image(
    "../figures/class_distribution.png",
    width: 90%
  ),
  caption: "Phân tích phân bố nhãn cảm xúc trong tập huấn luyện"
)
Sự mất cân bằng này đòi hỏi các kỹ thuật xử lý đặc biệt trong quá trình huấn luyện để đảm bảo mô hình không bị thiên lệch về các lớp có nhiều mẫu hơn.

== Thiết kế Dataset và DataLoader

=== Lớp SentimentDataset

Lớp SentimentDataset được thiết kế kế thừa từ torch.utils.data.Dataset để xử lý dữ liệu một cách hiệu quả:

```python
	class SentimentDataset(Dataset):
	    def __init__(self, df, tokenizer, max_len=120):
	        self.df = df
	        self.max_len = max_len
	        self.tokenizer = tokenizer
	
	    def __len__(self):
	        return len(self.df)
	
	    def __getitem__(self, index):
	        row = self.df.iloc[index]
	        text, label = self.get_input_data(row)
	
	        encoding = self.tokenizer.encode_plus(
	            text,
	            truncation=True,
	            add_special_tokens=True,
	            max_length=self.max_len,
	            padding='max_length',
	            return_attention_mask=True,
	            return_token_type_ids=False,
	            return_tensors='pt',
	        )
	
	        return {
	            'text': text,
	            'input_ids': encoding['input_ids'].flatten(),
	            'attention_masks': encoding['attention_mask'].flatten(),
	            'targets': torch.tensor(label, dtype=torch.long),
	        }
```

=== Tiền xử lý văn bản

Quá trình tiền xử lý văn bản được thực hiện theo quy trình đã được trình bày trong chương 3:

```python
	def get_input_data(self, row):
	    text = row['Sentence']
	    # Chuẩn hóa văn bản cơ bản
	    text = ' '.join(simple_preprocess(text))
	    # Phân đoạn từ tiếng Việt
	    text = ViTokenizer.tokenize(text)
	    label = self.labelencoder(row['Emotion'])
	    return text, label
```

=== Mã hóa nhãn

Hệ thống mã hóa nhãn được thiết kế để chuyển đổi nhãn cảm xúc từ dạng chuỗi sang dạng số:

```python
	def labelencoder(self, text):
	    emotion_mapping = {
	        'Enjoyment': 0, 'Disgust': 1, 'Sadness': 2,
	        'Anger': 3, 'Surprise': 4, 'Fear': 5, 'Other': 6
	    }
	    return emotion_mapping.get(text, 6)
```

== Cấu hình mô hình PhoBERT

=== Tải mô hình pre-trained

Mô hình PhoBERT-base được tải từ Hugging Face Hub và cấu hình cho bài toán phân loại:

```python
	phobert_model = "vinai/phobert-base"
	tokenizer = AutoTokenizer.from_pretrained(phobert_model, use_fast=False)
	num_labels = len(train_df['Emotion'].unique())
	
	model = AutoModelForSequenceClassification.from_pretrained(
	    phobert_model,
	    num_labels=num_labels
	).to(device)
```

=== Phân tích độ dài token

Phân tích phân bố độ dài token giúp xác định giá trị max_length tối ưu:

```python
	all_data = train_df.Sentence.tolist() + test_df.Sentence.tolist()
	all_data = [' '.join(simple_preprocess(text)) for text in all_data]
	encoded_text = [tokenizer.encode(text, add_special_tokens=True) for text in all_data]
	token_lens = [len(text) for text in encoded_text]
```

Kết quả phân tích cho thấy:

#figure(
  image(
    "../figures/token_length_analysis.png",
    width: 90%
  ),
  caption: "Phân tích độ dài token trong dữ liệu"
)
Dựa trên phân tích này, max_length=50 được chọn để cân bằng giữa việc bảo toàn thông tin và hiệu quả tính toán.

== Cấu hình huấn luyện


Các tham số huấn luyện được cấu hình dựa trên những nghiên cứu trước đó cho fine-tuning BERT:

```python
	training_args = TrainingArguments(
	    output_dir='/content/model',
	    logging_dir='/content/model/logs',
	    num_train_epochs=5,
	    learning_rate=2e-5,
	    per_device_train_batch_size=16,
	    per_device_eval_batch_size=16,
	    weight_decay=0.01,
	    eval_strategy="epoch",
	    save_strategy="epoch",
	    load_best_model_at_end=True,
	    metric_for_best_model="accuracy",
	    
	    # Tối ưu hóa
	    fp16=True,
	    dataloader_num_workers=2,
	    dataloader_pin_memory=True,
	    optim="adamw_torch",
	    group_by_length=True,
	    remove_unused_columns=False,
	    
	    # Logging
	    logging_steps=50,
	    logging_first_step=True,
	    
	    # Đánh giá
	    eval_accumulation_steps=10,
	    prediction_loss_only=False,
	)
```

=== Hàm đánh giá

Hàm compute_metrics được thiết kế để tính toán các chỉ số đánh giá trong quá trình huấn luyện:

```python
	    def compute_metrics(eval_pred):
	    import evaluate
	    import numpy as np
	
	    accuracy_metric = evaluate.load("accuracy")
	    f1_metric = evaluate.load("f1")
	
	    logits, labels = eval_pred
	    predictions = np.argmax(logits, axis=-1)
	
	    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
	    f1 = f1_metric.compute(predictions=predictions, references=labels, average='weighted')
	
	    return {
	        "accuracy": accuracy["accuracy"],
	        "f1": f1["f1"],
	    }
```

=== Data Collator

Data Collator tùy chỉnh được thiết kế để xử lý và tổ chức dữ liệu theo lô (batch) một cách hiệu quả trong quá trình huấn luyện. Thành phần này đóng vai trò quan trọng trong việc chuẩn bị dữ liệu đầu vào cho mô hình PhoBERT.

#figure(
  table(
    columns: 2,
    table.header(
      [*Thành phần*], [*Mô tả*]
    ),
    [input_ids], [Tensor chứa ID của các token đã được mã hóa],
    [attention_mask], [Tensor nhị phân xác định token thực và token đệm],
    [labels], [Tensor chứa nhãn cảm xúc đích cho việc huấn luyện]
  ),
  caption: "Hình 5.1: Các thành phần dữ liệu trong Data Collator"
)

Cài đặt Data Collator:

== Quá trình huấn luyện

=== Khởi tạo Trainer

Hugging Face Trainer được sử dụng để đơn giản hóa quá trình huấn luyện:

```python
	trainer = Trainer(
	    model=model,
	    args=training_args,
	    train_dataset=train_dataset,
	    eval_dataset=valid_dataset,
	    processing_class=tokenizer,
	    data_collator=data_collator,
	    compute_metrics=compute_metrics,
	)
```

=== Tối ưu hóa bộ nhớ

Trước khi bắt đầu huấn luyện, các kỹ thuật tối ưu hóa bộ nhớ được áp dụng:

```python
	import gc
	gc.collect()
	torch.cuda.empty_cache()
```

=== Thực hiện huấn luyện

Quá trình huấn luyện được thực hiện với monitoring chi tiết:

```python
	trainer.train()
```

Trong quá trình huấn luyện, các chỉ số sau được theo dõi:

- Training loss
- Validation accuracy
- Validation F1-score
- Learning rate decay

== Hệ thống đánh giá và trực quan hóa

=== Hàm dự đoán cảm xúc

Hàm dự đoán được thiết kế để xử lý văn bản đầu vào và trả về kết quả phân loại:

```python
	def predict_emotion(text, model, tokenizer, device):
	    id2label = {
	        0: 'Enjoyment', 1: 'Disgust', 2: 'Sadness',
	        3: 'Anger', 4: 'Surprise', 5: 'Fear', 6: 'Other'
	    }
	
	    # Tiền xử lý văn bản
	    processed_text = ' '.join(simple_preprocess(text))
	    processed_text = ViTokenizer.tokenize(processed_text)
	
	    # Token hóa
	    inputs = tokenizer.encode_plus(
	        processed_text,
	        truncation=True,
	        add_special_tokens=True,
	        max_length=120,
	        padding='max_length',
	        return_attention_mask=True,
	        return_token_type_ids=False,
	        return_tensors='pt',
	    )
	
	    # Dự đoán
	    inputs = {k: v.to(device) for k, v in inputs.items()}
	    model.eval()
	    with torch.no_grad():
	        outputs = model(input_ids=inputs['input_ids'],
	                       attention_mask=inputs['attention_mask'])
	        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
	        predicted_class = torch.argmax(predictions, dim=-1).item()
	
	    return predicted_class, predictions[0].tolist(), id2label[predicted_class]
```

=== Trực quan hóa kết quả

Hệ thống trực quan hóa bao gồm hai thành phần chính:

=== Confusion Matrix:

```python
	def plot_confusion_matrix(y_true, y_pred, labels):
	    cm = confusion_matrix(y_true, y_pred)
	    plt.figure(figsize=(10, 8))
	    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
	                xticklabels=labels, yticklabels=labels)
	    plt.title('Confusion Matrix')
	    plt.ylabel('True Label')
	    plt.xlabel('Predicted Label')
	    plt.show()
```

=== Phân bố xác suất:

```python
	def plot_probability_distribution(text, probabilities, labels):
	    plt.figure(figsize=(12, 6))
	    bars = plt.bar(labels, probabilities)
	    plt.title(f'Emotion Probabilities for: "{text}"')
	    plt.xlabel('Emotions')
	    plt.ylabel('Probability')
	    
	    for bar in bars:
	        height = bar.get_height()
	        plt.text(bar.get_x() + bar.get_width()/2., height,
	                f'{height:.2%}', ha='center', va='bottom')
	    plt.show()
```
== Kết luận chương

Chương này đã trình bày chi tiết quá trình triển khai mô hình PhoBERT cho bài toán phân loại cảm xúc văn bản tiếng Việt. Các thành phần chính bao gồm:

1. Chuẩn bị môi trường: Thiết lập môi trường phát triển với các thư viện cần thiết
2. Xử lý dữ liệu: Thiết kế quy trình xử lý dữ liệu hiệu quả với Dataset và DataLoader tùy chỉnh
3. Cấu hình mô hình: Tải và cấu hình PhoBERT cho bài toán phân loại
4. Huấn luyện: Thực hiện fine-tuning với các tham số được tối ưu hóa
5. Đánh giá: Xây dựng hệ thống đánh giá và trực quan hóa kết quả
