# Thai Handwriting Recognition Project ๔๑–๔๕ (Improved UI + Admin Login)

โปรเจกต์นี้เป็นเว็บแอปสำหรับงาน CS462: ทำนายลายมือเขียนเลขไทยจาก Canvas บน Browser  
คลาสที่ใช้คือ ๔๑, ๔๒, ๔๓, ๔๔, ๔๕

## สิ่งที่เพิ่มให้ในเวอร์ชันนี้
- ปรับ UI ให้ดูทันสมัยและสวยขึ้น
- เอา badge ด้านบนออกให้หน้าเรียบขึ้น
- เพิ่ม Admin Login
- Username admin = `admin`
- Password = `1234`
- รวมส่วน Collect Dataset ไว้ในหน้า Admin Dashboard
- หน้า Admin Dashboard สำหรับอัปโหลดโมเดล

## วิธีเปิดใน VS Code
```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

## URL สำคัญ
- User Page: `http://127.0.0.1:5000/`
- Admin Login: `http://127.0.0.1:5000/admin/login`
- Admin Dashboard: `http://127.0.0.1:5000/admin/dashboard`

## วิธี Train
1. เข้า `/admin/login` แล้วล็อกอินด้วย admin / 1234
2. เก็บรูป ๔๑–๔๕ ในหน้า Admin Dashboard
3. หยุด server ด้วย `Ctrl + C`
4. รัน `python train_model.py`
5. จะได้โมเดลที่ `models/thai_digit_model.keras`
