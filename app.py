import os
import cv2
import numpy as np
import base64
import json
from flask import Flask, request, jsonify
from waitress import serve

app = Flask(__name__)

TEMPLATE_JSON = 'omr_template.json'
TEMPLATE_IMG = 'omr_template_base.jpg'
RESULTS_DB = 'omr_results.json'

def load_db():
    if os.path.exists(RESULTS_DB):
        try:
            with open(RESULTS_DB, 'r', encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return {}

def save_db(data):
    with open(RESULTS_DB, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def align_with_sift(im_student, im_template):
    gray1 = cv2.cvtColor(im_student, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im_template, cv2.COLOR_BGR2GRAY)
    
    # 稍微放宽特征点数量，加快速度并增加对数字原图的宽容度
    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or len(des1) < 10 or des2 is None or len(des2) < 10:
        raise Exception("特征点过少，试卷可能极度模糊或未拍全")

    index_params = dict(algorithm=1, trees=5) 
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    # Lowe's ratio test 放宽到 0.75，适应更多噪点
    for m_n in matches:
        if len(m_n) != 2: continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) > 30:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None: raise Exception("变换矩阵计算失败，可能变形过于严重")
        h, w = im_template.shape[:2]
        return cv2.warpPerspective(im_student, M, (w, h))
    else:
        raise Exception(f"匹配点不足 ({len(good_matches)}/30)，请确保模板与试卷材质接近")

@app.route('/api/init_template', methods=['POST'])
def init_template():
    file = request.files['file']
    np_arr = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    h, w = image.shape[:2]
    max_w = 1200
    if w > max_w:
        ratio = max_w / float(w)
        image = cv2.resize(image, (max_w, int(h * ratio)))
    
    cv2.imwrite(TEMPLATE_IMG, image)
    _, buffer = cv2.imencode('.jpg', image)
    return jsonify({
        "status": "success",
        "image_base64": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}",
        "width": image.shape[1], "height": image.shape[0]
    })

@app.route('/api/extract_roi', methods=['POST'])
def extract_roi():
    data = request.json
    rois = data['rois']
    config = data.get('config', {"thresh": 0, "min_w": 12, "max_w": 100})
    
    if not os.path.exists(TEMPLATE_IMG):
        return jsonify({"status": "error", "message": "模板图丢失"}), 400
        
    image = cv2.imread(TEMPLATE_IMG)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 加入 CLAHE 抗反光/抗阴影
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    if int(config['thresh']) > 0:
        thresh = cv2.threshold(gray, int(config['thresh']), 255, cv2.THRESH_BINARY_INV)[1]
    else:
        # 优化自适应参数：blockSize 51 保证能圈住整个气泡，C=10 放宽对比度要求
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            51, 10
        )
    
    extracted_questions = []
    student_info_roi = None
    
    for roi in rois:
        if roi['type'] == 'info':
            student_info_roi = {"x": int(roi['x']), "y": int(roi['y']), "w": int(roi['w']), "h": int(roi['h'])}
            cv2.rectangle(image, (int(roi['x']), int(roi['y'])), (int(roi['x'])+int(roi['w']), int(roi['y'])+int(roi['h'])), (255, 165, 0), 2)
            continue

        start_q, q_count = int(roi['start_q']), int(roi['q_count'])
        direction = roi.get('direction', 'vertical')
        x, y, w, h = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])
        roi_thresh = thresh[y:y+h, x:x+w]
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        closed = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)
        
        cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bubbles = []
        for c in cnts:
            (bx, by, bw, bh) = cv2.boundingRect(c)
            ar = bw / float(bh)
            if bw >= int(config['min_w']) and bw <= int(config['max_w']) and 0.4 <= ar <= 4.0:
                bubbles.append((x + bx, y + by, bw, bh))
                
        if not bubbles: continue
            
        if direction == 'horizontal':
            bubbles = sorted(bubbles, key=lambda b: b[0])
            cols = []
            current_col = [bubbles[0]]
            for b in bubbles[1:]:
                if abs(b[0] - current_col[0][0]) < 20: 
                    current_col.append(b)
                else:
                    cols.append(sorted(current_col, key=lambda i: i[1]))
                    current_col = [b]
            cols.append(sorted(current_col, key=lambda i: i[1]))
            
            for i, col in enumerate(cols):
                if i >= q_count: break
                extracted_questions.append({
                    "question_num": start_q + i, "options_count": len(col),
                    "bubbles": [{"x": bx, "y": by, "w": bw, "h": bh} for (bx, by, bw, bh) in col]
                })
                for (bx, by, bw, bh) in col:
                    cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                    
        else:
            bubbles = sorted(bubbles, key=lambda b: b[1])
            rows = []
            current_row = [bubbles[0]]
            for b in bubbles[1:]:
                if abs(b[1] - current_row[0][1]) < 20: 
                    current_row.append(b)
                else:
                    rows.append(sorted(current_row, key=lambda i: i[0]))
                    current_row = [b]
            rows.append(sorted(current_row, key=lambda i: i[0]))
            
            for i, row in enumerate(rows):
                if i >= q_count: break
                extracted_questions.append({
                    "question_num": start_q + i, "options_count": len(row),
                    "bubbles": [{"x": bx, "y": by, "w": bw, "h": bh} for (bx, by, bw, bh) in row]
                })
                for (bx, by, bw, bh) in row:
                    cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                
    extracted_questions = sorted(extracted_questions, key=lambda q: q['question_num'])
    _, buffer = cv2.imencode('.jpg', image)
    return jsonify({
        "status": "success", "structure": extracted_questions, "student_info_roi": student_info_roi,
        "preview_image": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    })

@app.route('/api/save_template', methods=['POST'])
def save_template():
    data = request.json
    try:
        with open(TEMPLATE_JSON, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        save_db({}) 
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/get_template', methods=['GET'])
def get_template():
    if os.path.exists(TEMPLATE_JSON) and os.path.exists(TEMPLATE_IMG):
        try:
            with open(TEMPLATE_JSON, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image = cv2.imread(TEMPLATE_IMG)
            _, buffer = cv2.imencode('.jpg', image)
            img_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            
            return jsonify({
                "status": "success", 
                "data": data,
                "image_base64": img_b64,
                "width": image.shape[1],
                "height": image.shape[0]
            })
        except: pass
    return jsonify({"status": "not_found"})

@app.route('/api/toggle_flag', methods=['POST'])
def toggle_flag():
    filename = request.json.get('filename')
    db = load_db()
    if filename in db:
        db[filename]['flagged'] = not db[filename].get('flagged', False)
        save_db(db)
        return jsonify({"status": "success", "flagged": db[filename]['flagged']})
    return jsonify({"status": "error"})

@app.route('/api/analyze_single', methods=['POST'])
def analyze_single():
    file = request.files['file']
    filename = file.filename
    force_recalc = request.form.get('force', 'false') == 'true'
    
    db = load_db()
    
    if not force_recalc and filename in db:
        return jsonify({"status": "success", "result": db[filename]})

    try:
        with open(TEMPLATE_JSON, 'r', encoding='utf-8') as f:
            template_data = json.load(f)
        template_img = cv2.imread(TEMPLATE_IMG)
        
        np_arr = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # --- 1. SIFT 工业对齐 ---
        aligned_color = align_with_sift(image, template_img)
        aligned_gray = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2GRAY)
        
        # --- 2. 图像预处理增强 (抗阴影、抗反光) ---
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        aligned_gray = clahe.apply(aligned_gray)
        
        # --- 3. 局部自适应二值化 (抗透背) ---
        thresh = cv2.adaptiveThreshold(
            aligned_gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            51, 10
        )

        score = 0
        student_details = []
        questions = template_data.get('questions', [])
        info_roi = template_data.get('student_info_roi')

        info_b64 = None
        if info_roi:
            ix, iy, iw, ih = info_roi['x'], info_roi['y'], info_roi['w'], info_roi['h']
            # 安全防越界保护
            h_img, w_img = aligned_color.shape[:2]
            ix, iy = max(0, ix), max(0, iy)
            iw, ih = min(iw, w_img - ix), min(ih, h_img - iy)
            
            if iw > 0 and ih > 0:
                info_crop = aligned_color[iy:iy+ih, ix:ix+iw]
                _, buf = cv2.imencode('.jpg', info_crop)
                info_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"

        for q in questions:
            q_num = q['question_num']
            bubbles = q['bubbles']
            correct_idx = q['correct_index']
            weight = q['weight']
            options_labels = q.get('labels', ["A", "B", "C", "D", "E", "F", "G", "H"])
            
            max_pixels = 0
            selected_idx = -1

            for j, b in enumerate(bubbles):
                x, y, w, h = b['x'], b['y'], b['w'], b['h']
                roi = thresh[y+2:y+h-2, x+2:x+w-2]
                total_pixels = cv2.countNonZero(roi)
                
                # 最低填涂占比判定
                roi_area = roi.shape[0] * roi.shape[1]
                if total_pixels > max_pixels and total_pixels > (roi_area * 0.15):
                    max_pixels = total_pixels
                    selected_idx = j

            is_correct = (selected_idx == correct_idx)
            if is_correct: score += weight
            
            for j, b in enumerate(bubbles):
                x, y, w, h = b['x'], b['y'], b['w'], b['h']
                if j == selected_idx:
                    cv2.rectangle(aligned_color, (x, y), (x+w, y+h), (0, 255, 0) if is_correct else (0, 0, 255), 3)
                elif j == correct_idx and not is_correct:
                    cv2.rectangle(aligned_color, (x, y), (x+w, y+h), (255, 191, 0), 2)
                    
            student_details.append({
                "question": q_num,
                "selected": options_labels[selected_idx] if selected_idx != -1 and selected_idx < len(options_labels) else "未填",
                "is_correct": is_correct
            })

        _, buffer = cv2.imencode('.jpg', aligned_color)
        result_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

        result_obj = {
            "filename": filename,
            "score": score,
            "status": "success",
            "details": student_details,
            "image_base64": result_b64,
            "info_base64": info_b64,
            "flagged": False
        }
        
        db[filename] = result_obj
        save_db(db)

        return jsonify({"status": "success", "result": result_obj})

    except Exception as e:
        # 将后端异常完整穿透，方便你排查是真的没对齐，还是其他逻辑错乱
        print(f"[后端错误] 文件 {filename} 批改失败: {str(e)}")
        return jsonify({"status": "error", "message": f"解析异常: {str(e)}", "filename": filename})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    print("--------------------------------------------------")
    print(" 智能阅卷系统 V6.6 (坚如磐石底座版) 已启动 ")
    print(" 核心升级：修复局部代码替换缩进隐患，重调自适应宽容度")
    print("--------------------------------------------------")
    serve(app, host='127.0.0.1', port=5000)