= Kiến trúc mô hình REFRAG

Trong phần này, em trình bày kiến trúc chi tiết của REFRAG theo cách tiếp cận được mô tả trong bài báo gốc. REFRAG được xây dựng dựa trên một mô hình ngôn ngữ chỉ có decoder (ví dụ LLaMA-2-7B hoặc LLaMA-3.2-3B) kết hợp với một encoder gọn nhẹ (ví dụ RoBERTa-large). Để thể hiện rõ ràng cơ chế hoạt động, em tập trung vào một lượt truy vấn–truy hồi đơn lẻ; trường hợp hội thoại nhiều lượt sẽ được mở rộng ở các phần sau.

Ký hiệu mô hình ngôn ngữ chỉ có decoder là $M_(upright(d e c))$ và encoder là $M_(upright(e n c))$. Cho một chuỗi đầu vào gồm $T$ token $x_(1), x_(2), ..., x_(T)$, em giả sử $q$ token đầu là phần truy vấn/chính (ví dụ câu hỏi) và $s$ token cuối là phần ngữ cảnh (các đoạn truy hồi trong RAG), sao cho $q + s = T$.

== Tổng quan kiến trúc

 Kiến trúc tổng thể của mô hình RAG gồm:
  -  *$M_(upright(d e c))$*: một LLM chỉ có decoder (ví dụ Llama-3.2-3B), chịu trách nhiệm sinh câu trả lời.
  -  *$M_(upright(e n c))$*: một mô hình mã hoá (ví dụ RoBERTa-large) dùng để tạo embedding cho từng đoạn ngữ cảnh.
  -  *Tầng chiếu $phi.alt $*: một lớp tuyến tính (hoặc MLP nhỏ) chiếu embedding đoạn vào không gian embedding token của decoder.
  -  *Chính sách RL mở rộng đoạn*: một mạng nhỏ (transformer hoặc MLP) hoạt động trên các embedding đoạn để quyết định đoạn nào cần mở rộng ra token đầy đủ.

 Dòng xử lý cơ bản như sau:
  +  Truy vấn (câu hỏi) $x_(1 : q)$ được token hoá và đưa qua embedding token chuẩn của LLM, thu được các embedding $e_(1), ..., e_(q)$.
  +  Ngữ cảnh $x_(q + 1 : T)$ được chia thành các đoạn $C_(i)$ cố định, mỗi đoạn có độ dài $k $ token.
  +  Mỗi đoạn $C_(i)$ được mã hoá bằng encoder $M_(upright(e n c))$ để thu được embedding đoạn $c_(i)$.
  +  Các embedding đoạn $c_(i)$ được chiếu bằng $phi.alt $ sang embedding “giả token” $tilde(c)_(i)$ có cùng chiều với embedding token của decoder.
  +  Chuỗi đầu vào của decoder được hình thành bằng cách nối $e_(1), ..., e_(q)$ với các embedding $tilde(c)_(1), ..., tilde(c)_(L)$ (hoặc tổ hợp token + embedding nếu có nén chọn lọc).
  +  Decoder $M_(upright(d e c))$ sinh ra chuỗi trả lời tự hồi quy $y $.

 Mỗi $tilde(c)_(i)$ đóng vai trò như một “siêu token” đại diện cho cả một đoạn ngữ cảnh dài $k $ token. Khi $s gt.double q $ (tức phần ngữ cảnh áp đảo phần truy vấn – rất phổ biến trong RAG), việc thay thế $s $ token ngữ cảnh bằng $L = s /k $ embedding đoạn giúp giảm số vị trí mà decoder phải xử lý.

== Chia đoạn và embedding theo đoạn

 Phần ngữ cảnh $x_(q + 1 : T)$ được chia thành các đoạn con có độ dài cố định $k $: $  L : = s/k,  $
 Ký hiệu: $  C_(i) = { x_(q + k(i - 1) + 1), ..., x_(q + k i) }, i = 1, ..., L .  $
 Mỗi đoạn $C_(i)$ được mã hoá bởi encoder: $  c_(i) = M_(upright(e n c))(C_(i)).  $
 Vì $M_(upright(e n c))$ là một mô hình gọn nhẹ (ví dụ RoBERTa-large $tilde 3 5 5 $M tham số), các đoạn $C_(i)$ có thể được xử lý song song mà không cần attention chéo giữa các đoạn; điều này quan trọng vì embedding đoạn có thể được *tính trước và cache* trong hệ thống RAG thực tế.

Để sử dụng trong decoder, ta chiếu $c_(i)$ sang không gian embedding token $d $ chiều của LLM: $  tilde(c)_(i) = phi.alt(c_(i)) in RR^(d),  $
 trong đó $phi.alt $ là một tầng tuyến tính học được (hoặc MLP nhỏ).

 Ký hiệu $e_(i)$ là embedding token của $x_(i)$ với $i = 1, ..., q $. Mô hình RAG tạo đầu vào nén cho decoder bằng cách nối: $  y tilde M_(upright(d e c)) e_(1), ..., e_(q), tilde(c)_(1), ..., tilde(c)_(L) .  $

 Trong nhiều kịch bản RAG, độ dài ngữ cảnh $s $ lớn hơn rất nhiều so với độ dài truy vấn $q $ ($s gt.double q $). Khi đó, số vị trí mà decoder phải xử lý giảm từ $  T = q + s  $ xuống xấp xỉ $  T ' = q + L = q + s/k.  $ Nếu $q $ tương đối nhỏ so với $s $, có thể coi $T ' approx s/k$, tức là rút ngắn chuỗi khoảng một hệ số $k $.

== Ví dụ trực quan về kiến trúc

Để minh hoạ, xét tình huống câu hỏi: "_Ai là tổng thống Hoa Kỳ?_" và các đoạn truy hồi chứa thông tin như:
  - "Donald Trump is the President of the United States. He assumed office on January 20, 2025, making him the 47th President of the United States."
  - Các đoạn khác nói về các đời tổng thống trước đó, các sự kiện khác, v.v.

Trong thiết kế mô hình REFRAG:
  - Câu hỏi $x_(1:q)$ = "_Ai là tổng thống Hoa Kỳ?_" được tokenize thành khoảng 10 token, tạo embedding $e_(1), ..., e_(q)$ qua embedding token của LLM.
  - Các đoạn truy hồi (tổng $s$ token) được chia thành các chunk có kích thước $k$ (ví dụ $k = 16$ token). Với đoạn đầu tiên về Donald Trump, có thể chia thành 2 chunk: chunk thứ nhất chứa thông tin về tên và vị trí hiện tại, chunk thứ hai chứa ngày nhậm chức và thứ tự.
  - Mỗi chunk được mô hình encoder RoBERTa mã hoá thành vector $c_(i)$ (ví dụ 768 chiều). Chunk 1 sẽ có embedding $c_(1)$ chứa ngữ nghĩa chính về "Donald Trump is President", chunk 2 có embedding $c_(2)$ chứa ngữ nghĩa về ngày tháng và thứ tự.
  - Các embedding đoạn $c_(1), c_(2), ...$ được chiếu qua ma trận $phi.alt$ sang không gian 4096 chiều (cùng chiều số chiều ẩn của LLM) để tạo các "siêu token" giả $tilde(c)_(1), tilde(c)_(2), ...$.
  - Chuỗi đầu vào cho decoder bây giờ gồm: các token embedding của câu hỏi ($e_(1), ..., e_(q)$, khoảng 10 token) + các siêu token ($tilde(c)_(1), tilde(c)_(2), ..., tilde(c)_(L)$, khoảng $L = s/k$ siêu token). Nếu $s = 160$ token ngữ cảnh và $k = 16$, thì $L = 10$ siêu token, tổng độ dài input chỉ còn 20 token thay vì 170 token.
  - Decoder LLaMA sinh câu trả lời "_Donald Trump là tổng thống Hoa Kỳ hiện tại, nhậm chức ngày 20 tháng 1 năm 2025._" dựa trên chuỗi input nén này. Nếu chính sách RL nhận định rằng chunk 1 chứa thông tin quan trọng nhất, nó có thể "mở rộng" $tilde(c)_(1)$ thành các token gốc để cho phép decoder tiếp cận chi tiết hơn.

 Các chi tiết về quy trình huấn luyện (CPT, nhiệm vụ tái dựng, giáo trình, và nén chọn lọc bằng RL) sẽ được trình bày trong chương tiếp theo.
