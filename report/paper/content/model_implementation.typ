= Phương pháp

 Trong phần này, em trình bày chi tiết cách huấn luyện và sử dụng mô hình REFRAG trên nền một mô hình chỉ có decoder (ví dụ Llama-3.2-3B). Mục tiêu của là:
  -  Dạy cho decoder _hiểu_ embedding đoạn như một dạng “token nén” có ý nghĩa ngữ cảnh.
  - Đảm bảo khi thay đoạn token gốc bằng embedding đoạn, phân phối sinh $p(y divides "context")$ của mô hình không bị lệch nhiều.
  -  Học một cơ chế nén chọn lọc sử dụng RL để đạt cân bằng giữa độ trễ và độ chính xác.

Để làm được điều này, em sử dụng ba thành phần chính:
  +  *Tiếp tục tiền huấn luyện* (continual pre-training, viết tắt CPT) với nhiệm vụ dự đoán đoạn kế tiếp.
  +  *Nhiệm vụ tái dựng* (reconstruction) để căn chỉnh encoder + tầng chiếu với decoder.
  +  *Học theo giáo trình* (curriculum learning) và *chính sách RL* cho nén chọn lọc.

== Tiếp tục tiền huấn luyện với dự đoán đoạn kế tiếp

 Mục tiêu của CPT là: khi thay ngữ cảnh dạng token $x_(1 : s)$ bằng embedding đoạn $tilde(c)_(1),...,tilde(c)_(L)$, mô hình vẫn có thể dự đoán đoạn tiếp theo $x_(s + 1 : s + o)$ với chất lượng tương đương.

=== Thiết lập bài toán

 Mỗi mẫu huấn luyện gồm $T = s + o $ token: $  x_(1), x_(2), ..., x_(s), x_(s + 1), ..., x_(s + o),  $ trong đó:
  - $x_(1 : s)$ đóng vai trò ngữ cảnh,
  - $x_(s + 1 : s + o)$ là đoạn nối tiếp cần dự đoán.

	Chia $x_(1 : s)$ thành $L = s /k $ đoạn $C_(i)$ như đã mô tả ở Chương 2, mã hoá thành $c_(i)$, chiếu thành $tilde(c)_(i)$, và tạo chuỗi đầu vào nén: $  upright(I n p u t) = tilde(c)_(1), ..., tilde(c)_(L)  $ (hoặc $(e_(1),...,e_(q),tilde(c)_(1),...,tilde(c)_(L))$ nếu có truy vấn ở đầu).

=== Hàm mục tiêu

 Hàm mất mát CPT là cross-entropy chuẩn trên đoạn đầu ra $x_(s + 1 : s + o)$: $  cal(L)_("CPT") = - sum_(t = s + 1)^(s + o) log p_(theta)(x_(t) divides upright(I n p u t), x_(s + 1 : t - 1)).  $
 Ở đây, $theta $ là tham số kết hợp của:
  -  encoder *$M_(upright(e n c))$*,
  -  tầng chiếu *$phi.alt $*,
  -  và decoder *$M_(upright(d e c))$*.

 Trong giai đoạn CPT:
  -  Khởi tạo *$M_(upright(d e c))$* từ một LLM đã được tiền huấn luyện (ví dụ Llama-3.2-3B).
  -  Khởi tạo *$M_(upright(e n c))$* từ một encoder có sẵn (thường là mô hình bi-encoder trong pipeline truy hồi, ví dụ RoBERTa).
  -  Khởi tạo *$phi.alt $* ngẫu nhiên.
 Sau đó, trong CPT, cho phép tất cả ba thành phần được cập nhật (nhưng có thể dùng learning rate nhỏ cho decoder để tránh quên kiến thức ngôn ngữ gốc).

== Tái dựng

 Mặc dù CPT giúp mô hình học sử dụng embedding đoạn để dự đoán đoạn tiếp theo, nó chưa đảm bảo rằng embedding đoạn chứa đủ thông tin để “tái hiện” lại chuỗi gốc $x_(1 : s)$. Nếu embedding chứa quá ít thông tin, mô hình sẽ phải dựa nhiều vào “bộ nhớ tham số” (parametric memory), dẫn đến khả năng sử dụng ngữ cảnh bị hạn chế.

=== Thiết lập

 Trong nhiệm vụ tái dựng, cho đầu vào là $x_(1 : s)$, chia thành các đoạn $C_(1), ..., C_(L)$, rồi: $  c_(i) & = M_(upright(e n c))(C_(i)), \ tilde(c)_(i) & = phi.alt(c_(i)).  $
 Decoder nhận chuỗi $(tilde(c)_(1),...,tilde(c)_(L))$ và được yêu cầu tái dựng _chính_ chuỗi $x_(1 : s)$.

=== Hàm mất mát tái dựng

 Hàm mất mát: $  cal(L)_("recon") = - sum_(t = 1)^(s) log p_(theta)(x_(t) divides tilde(c)_(1),...,tilde(c)_(L), x_(1 : t - 1)^("(pred)")),  $
 trong đó $x_(1 : t - 1)^("(pred)")$ là các token đã sinh trước đó trong quá trình tái dựng.

Trong giai đoạn tái dựng:
  -  Đóng băng toàn bộ tham số của decoder *$M_(upright(d e c))$*.
  -  Chỉ cập nhật *$M_(upright(e n c))$* và *$phi.alt $*.

Ý tưởng là: decoder vốn đã học rất tốt phân phối ngôn ngữ $p(x)$ trong tiền huấn luyện; nhiệm vụ của encoder và tầng chiếu là “nén” thông tin của $x_(1 : s)$ vào các embedding đoạn sao cho, khi decoder đọc các embedding này, nó có đủ thông tin để tái dựng chuỗi gốc.

== Học theo giáo trình

 Nếu ngay lập tức yêu cầu mô hình tái dựng một chuỗi dài $s $ từ $L $ embedding đoạn, bài toán có thể quá khó, đặc biệt khi $k $ (độ dài đoạn) lớn. Không gian các chuỗi khả dĩ dài $k L $ token có khoảng $V^(k L)$ khả năng (với $V $ là kích thước từ vựng). Do đó, ta áp dụng học theo giáo trình.

=== Giáo trình cho nhiệm vụ tái dựng

 Em sắp xếp các bước:

  *Bước 1:* Chỉ tái dựng một đoạn duy nhất.  Encoder nhận $C_(1)$, sinh $c_(1)$, chiếu thành $tilde(c)_(1)$, và decoder tái dựng $x_(1 : k)$.

  *Bước 2:* Tái dựng hai đoạn.  Encoder nhận $(C_(1), C_(2))$, sinh $(tilde(c)_(1),tilde(c)_(2))$, decoder tái dựng $x_(1 : 2 k)$.

  *Bước 3:* Tăng dần số đoạn lên 4, 8, … cho đến $L $ (tương ứng với $s $ đầy đủ).

 Trong quá trình huấn luyện, minibatch của em là hỗn hợp các ví dụ dễ/khó:
  -  Giai đoạn đầu: tỉ lệ ví dụ “1 đoạn, 2 đoạn” cao.
  -  Giai đoạn sau: tỉ lệ ví dụ “nhiều đoạn” tăng lên, ví dụ 50% mẫu dùng $L $ đoạn, phần còn lại là các nhiệm vụ ngắn hơn để tránh mô hình bị “quá sức”.

=== Giáo trình cho CPT

 Với CPT, áp dụng logic tương tự:
  -  Bắt đầu với ngữ cảnh ngắn ($s $ nhỏ, ít đoạn).
  -  Tăng dần $s $ để CPT học dần cách sử dụng embedding đoạn cho ngữ cảnh dài hơn.

 Thực nghiệm cho thấy nếu không áp dụng giáo trình (tức huấn luyện trực tiếp trên ngữ cảnh dài ngay từ đầu), cả nhiệm vụ tái dựng và CPT đều đạt perplexity rất kém, đặc biệt khi $k >= 1 6 $. Khi áp dụng giáo trình, mô hình hội tụ ổn định hơn và tận dụng embedding đoạn tốt hơn.

== Huấn luyện và tinh chỉnh trên tác vụ hạ nguồn

 Sau khi hoàn thành tái dựng và CPT, ta sử dụng mô hình REFRAG như một thành phần cho các tác vụ hạ nguồn (QA, RAG, hội thoại, tóm tắt). Quy trình:
  +  *Khởi tạo*:  Sử dụng *$M_(upright(d e c))$*, *$M_(upright(e n c))$*, *$phi.alt $* đã huấn luyện từ CPT + tái dựng.
  +  *SFT*:  Huấn luyện với các cặp (ngữ cảnh truy hồi + câu hỏi, đáp án) theo định dạng hội thoại (ví dụ ``<s> INST ... /INST ...`` trong LLaMA), trong đó phần ngữ cảnh được xử lý bởi RAG theo embedding đoạn.
  +  *Nén chọn lọc*:  Trong giai đoạn này,  có thể:
    -  Giữ chính sách RL cố định (chỉ dùng inference-time policy).
    -  Hoặc tiếp tục tinh chỉnh chính sách RL song song với SFT, dùng reward là độ chính xác hoặc log-likelihood trên nhiệm vụ hạ nguồn.

== Nén chọn lọc bằng RL

=== Bài toán tối ưu

Có $L $ đoạn, muốn chọn $T ' = floor.l p L floor.r $ đoạn để mở rộng. Cấu hình lựa chọn $l = {l_(1),...,l_(T ')} subset.eq[L ]$ sẽ quyết định chuỗi đầu vào: $  E(x,l) = {E_(1),...,E_(L)},  $ trong đó: $  E_(i) = cases(tilde(c)_(i), & i in.not l,, {e_(q + k(i - 1)+ 1), ..., e_(q + k i)}, & i in l .)  $

Cho một nhiệm vụ cụ thể (ví dụ dự đoán đoạn kế tiếp)

Định nghĩa reward: $  r(x,l) = - 1/o sum_(t = s + 1)^(s + o) log p_(theta) x_(t) divides E(x,l), x_(s + 1 : t - 1),  $
tức là _âm log-perplexity trung bình_ trên đoạn đầu ra. 

Bài toán: $  max_(l subset.eq[L ]) & r(x, l) \ "s.t." & | l | = T '.  $

=== Thiết kế chính sách

Dùng một transformer nhỏ $g_(theta)$ trên các embedding đoạn ${c_(i)}$: $  s = g_(theta)(c_(1),...,c_(L)) in RR^(L),  $ trong đó $s_(i)$ là logit cho đoạn $i $. Ở mỗi bước $t = 1,...,T '$, chính sách chọn một đoạn mới: $  pi_(theta)(l_(t) = i divides x, l_(1 : t - 1)) = frac(exp(s_(i) - n_(i)), sum_(j = 1)^(L) exp(s_(j) - n_(j))),  $
 với: $  n_(i) = cases(+ infinity, & "nếu" i in {l_(1),...,l_(t - 1)},, 0, & "ngược lại.")  $ Tức là một dạng softmax với mask các vị trí đã chọn, đảm bảo “không chọn trùng”.

=== Huấn luyện RL

Đối với mỗi batch mẫu $(x, l_(1 : T '))$ sinh bởi chính sách, tính reward $r(x,l)$ và tối ưu hàm mục tiêu RL: $  J(theta) = bb(E)_(l tilde pi_(theta)(dot divides x))[r(x, l)].  $
 Gradient ước lượng bằng REINFORCE: $  nabla_(theta) J(theta) approx 1/B sum_(b = 1)^(B)(r(x^((b)), l^((b))) - b) nabla_(theta) log pi_(theta) l^((b)) divides x^((b)),  $
 với $b $ là baseline (ví dụ trung bình reward trong batch) để giảm phương sai.

Trong thực nghiệm, để _giảm chi phí_:
  -  Tái sử dụng cùng một vector logit $s $ tại mọi bước chọn $l_(t)$ (không cập nhật chiều ẩn sau mỗi lần chọn).
  -  Chỉ huấn luyện $g_(theta)$ (chính sách), giữ cố định $M_(upright(e n c))$ và $M_(upright(d e c))$, giúp RL ổn định hơn.

=== Heuristic so sánh
Xem xét một số heuristic:
  -  *Random*: chọn ngẫu nhiên $T '$ đoạn.
  -  *Perplexity-asc*: dùng LLM nền để tính perplexity từng đoạn, sau đó mở rộng các đoạn có perplexity _cao_ (cho rằng chúng khó dự đoán, nên cần giữ token đầy đủ).
  -  *Perplexity-desc*: ngược lại, mở rộng đoạn có perplexity _thấp_ (cho rằng chúng chứa thông tin rõ ràng, dễ “dùng” hơn).
