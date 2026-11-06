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
1. BoW + LogReg (tự code)           1.0   0.711      0.831        0.857   
2. BoW + MLP (deep learning)        1.0   0.835      0.910        0.947   
4a. LLM nhỏ zero-shot               1.0   0.950      0.974        0.974   
4b. LLM lớn zero-shot               1.0   0.950      0.974        0.974   
4c. LLM nhỏ few-shot                1.0   0.950      0.974        0.974   

                              train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)        12.014      51892.489       0.016   
2. BoW + MLP (deep learning)      1.964     151762.570       0.056   
4a. LLM nhỏ zero-shot             0.000          0.653    1234.650   
4b. LLM lớn zero-shot             0.000          0.674    1455.372   
4c. LLM nhỏ few-shot              0.000          0.727    1289.361   

                              p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)          0.027         0.000  
2. BoW + MLP (deep learning)       0.069         0.000  
4a. LLM nhỏ zero-shot           2271.229         0.010  
4b. LLM lớn zero-shot           1762.532         0.105  
4c. LLM nhỏ few-shot            1849.028         0.020  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
