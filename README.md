# PDF Question Answering (PDF QA)

PDF QA là một dự án dựa trên Python được thiết kế để tạo điều kiện thuận lợi cho việc trả lời câu hỏi trên các tài liệu PDF. Dự án này sử dụng các mô hình NLP tiên tiến và cơ sở dữ liệu vector để trả lời các câu hỏi từ người dùng về nội dung của tài liệu.

---

## Các bước thực hiện  
1. **Hỗ trợ các loại file**:  
   - PDF (bao gồm cả các file scanned).  
   - Docx.  
   - TXT.  

2. **Trích xuất văn bản**:  
   - **PyMuPDF** được sử dụng để trích xuất văn bản từ file PDF.  
   - **python-docx** được sử dụng để trích xuất văn bản từ file DOCX.  
   - **EasyOCR** được sử dụng để trích xuất văn bản từ hình ảnh có trong các file và xử lý các file PDF đã quét (scanned).  

3. **Chia tách văn bản**:  
   - Sau khi trích xuất văn bản, không thực hiện tách chuỗi theo dấu chấm mà theo **mô hình BAAI/bge-small-en-v1.5**, giúp cải thiện chất lượng phân đoạn văn bản.

4. **Chuyển đổi văn bản thành vector**:  
   - Văn bản được chuyển đổi thành vector sử dụng mô hình **all-MiniLM-L6-v2**.  
   - Các vector này được lưu trữ trong cơ sở dữ liệu vector **Chroma** để tiện việc tìm kiếm và truy xuất.

5. **Trả lời câu hỏi**:  
   - Khi nhận được câu hỏi từ người dùng, mô hình sẽ tìm kiếm các tài liệu có liên quan trong cơ sở dữ liệu vector.  
   - Câu hỏi cùng với tài liệu liên quan sẽ được đưa vào mô hình **gemini-1.5-pro-latest** để tạo ra câu trả lời cuối cùng.
   - Với các câu hỏi không có trong tài liệu liên quan, mô hình sẽ sinh ra câu trả lời dựa trên kiến thức của chính nó.
   - Prompt :
     ```bash
     Bạn sẽ trả lời câu hỏi của người dùng dựa trên nội dung và lịch sử trò chuyện sau: 
     Thông tin:
     {context}.

     Nếu không tìm thấy thông tin phù hợp hãy:
     - Thông báo cho người dùng biết bạn không tìm thấy thông tin.
     - Đưa ra câu trả lời dựa trên kiến thức nền tảng của bạn.
     Hãy giữ câu trả lời ngắn gọn và trả lời bằng tiếng Việt.
     Câu hỏi: {question}

     Câu trả lời:
     ```

![ss drawio](https://github.com/user-attachments/assets/89b1c13d-695d-4e77-8b4e-741fecfaf809)

---

## Hướng dẫn cài đặt và chạy

### Cài đặt các thư viện phụ thuộc

1. **Cài đặt các phụ thuộc** từ file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
