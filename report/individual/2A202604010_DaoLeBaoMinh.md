# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604010
- **Họ và tên:** Đào Lê Bảo Minh
- **SEED** (in ở ô đầu notebook): 6772

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*

precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.711      0.831        0.857   
2. BoW + MLP (deep learning)         1.000   0.845      0.916        0.947   
3a. Encoder (MiniLM) + kNN           0.856   0.918      0.886        0.950   
3b. Encoder (MiniLM) + LogReg        1.000   0.928      0.963        0.974   
3c. Encoder (MiniLM) + MLP head      0.968   0.938      0.953        0.974   
4a. LLM nhỏ zero-shot                1.000   0.950      0.974        0.974   
4b. LLM lớn zero-shot                1.000   0.950      0.974        0.974   
4c. LLM nhỏ few-shot                 1.000   0.900      0.947        0.947   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.952      92240.305       0.016   
2. BoW + MLP (deep learning)         1.244     149343.409       0.048   
3a. Encoder (MiniLM) + kNN           0.978       2922.259       6.552   
3b. Encoder (MiniLM) + LogReg        2.240       4770.350       2.664   
3c. Encoder (MiniLM) + MLP head      1.371       5843.965       2.897   
4a. LLM nhỏ zero-shot                0.000          0.810    1146.203   
4b. LLM lớn zero-shot                0.000          0.723    1263.666   
4c. LLM nhỏ few-shot                 0.000          0.841    1124.485   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.028         0.000  
2. BoW + MLP (deep learning)          0.079         0.000  
3a. Encoder (MiniLM) + kNN           10.457         0.000  
3b. Encoder (MiniLM) + LogReg         3.106         0.000  
3c. Encoder (MiniLM) + MLP head       4.670         0.000  
4a. LLM nhỏ zero-shot              1851.242         0.010  
4b. LLM lớn zero-shot              1568.236         0.105  
4c. LLM nhỏ few-shot               1685.098         0.020  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

- Chạy lại với `lr = 0.001` và `lr = 50`. Dự đoán trước hình dạng đường loss của **từng** trường hợp.
  predict("ml_lr", "lr=50 thì loss bùng nổ, càng lúc càng lớn, tốc độ tăng rất nhanh.")
  lr=50 loss không nổ mà vụt một phát xuống sát 0.003. em tưởng sẽ bị gradient exploding nhưng không. em không hiểu

- (2) MLP sẽ hơn, bằng hay kém logistic regression ở Phần 1 về F1?
  predict("dl_curve", "(2) tốt hơn logistic ở phần 1 do mạng sâu học được quan hệ phi tuyến.")
  Em tưởng có tầng ẩn thì tốt hơn, nhưng thực tế thì MLP với đầu vào là embedding thì không quá khá biệt, thậm chí điểm thấp hơn so với không dùng tầng ẩn.

- Truy vấn bên dưới **không chứa** các từ spam quen thuộc như "free", "win", "prize". Dự đoán: kết quả gần nhất là spam hay ham?
  predict("emb_search", "ham")
  Ban đầu em nghĩ lách được giống BoW, nhưng vì embedding học được ngữ nghĩa nên có thể so sánh được tương đồng với các câu spam tương tự.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

🎯 Đề #5: Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin.

Em nhờ AI sinh 4 tin nhắn theo yêu cầu đề bài, sau đó copy đoạn code ở phía trên cho việc dự đoán BoW/Embedding+LogReg để tính predictions.
Giải thích: BoW cho False hết, vì hai tin đầu spam nhưng không có từ spam nào trong từ điểm bị trigger. Hai tin sau thì chỉ có từ win và free trigger xuất hiện một lần, bù lại các từ ham lấn át.
Emb cho preds gần giống hệt, ngoại trừ câu đầu tiên. Em nghĩ với bộ test lớn hơn thì khác biệt thể hiện rõ hơn, nhưng về cơ bản khi tokenize tin nhắn thì đa số các từ vô nghĩa không có trong dữ liệu học, dẫn đến mô hình có xu hướng "không hiểu tin nhắn" nhiều hơn.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

Chủ yếu các phần em đều thử implement trước, trừ khi bí lắm mới nhờ AI. Gom đúng phần context cần thiết trước khi hỏi AI, sau đó sửa lại tên biến cho đúng. Chỗ nào quên API hoặc hiểu sai thì em nhờ AI trợ giúp cho em nhớ.
