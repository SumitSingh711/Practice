from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
# from langchain_community.document_loaders import loader

# loader = PyMuPDFLoader(
#     "dl-curriculum.pdf")

text = """class PDFAnnotator:
    def __init__(self):
        self.robot_info_path = os.path.join(app_dir, "robot_info.json")
        self.app_dir = app_dir
        self.page_data = None
        self.image_name = None
        self.page_id = None

    def to_pdf_coords(self, x, y, image_w, image_h, pdf_w, pdf_h):
        pdf_x = pdf_w-(x / image_w) * pdf_w
        pdf_y = pdf_h-(y / image_h) * pdf_h
        return pdf_y, pdf_x

    def to_pdf_rect(self, x1, y1, x2, y2, image_w, image_h, pdf_w, pdf_h):
        pdf_x1, pdf_y1 = self.to_pdf_coords(x1, y1, image_w, image_h, pdf_w, pdf_h)
        pdf_x2, pdf_y2 = self.to_pdf_coords(x2, y2, image_w, image_h, pdf_w, pdf_h)

        xmin, xmax = sorted([pdf_x1, pdf_x2])
        ymin, ymax = sorted([pdf_y1, pdf_y2])

        return (xmin, ymin, xmax, ymax)

    def json_loader_to_pdf(self, pdf_path, image_path, output_path=None):
        try:
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            image_name, page_id = os.path.splitext(os.path.basename(image_path))[0].split('-')

            with open(self.robot_info_path, "r") as f:
                all_data = json.load(f)

            if image_name not in all_data or page_id not in all_data[image_name]:
                print(f"No data found for {image_name} - {page_id}")
                return None

            self.page_data = all_data[image_name][page_id]
            self.image_name = image_name
            self.page_id = page_id


            doc = fitz.open(pdf_path)
            page = doc[0]
            pdf_w, pdf_h = page.rect.width, page.rect.height
            print(f'page_rotation: {page.rotation}')

            print(f'pdf_w: {pdf_w}, pdf_h: {pdf_h}')
            doc.close()

            packet = BytesIO()

            if pdf_w > pdf_h:
                c = canvas.Canvas(packet, pagesize=(pdf_w, pdf_w))
            else:
                c = canvas.Canvas(packet, pagesize=(pdf_h, pdf_h))
            scale_factor = pdf_w / image_width 
            
            for group_name, group_info in self.page_data.items():
                if group_name in ["Table width", "Panel width", "Number of sub-rows"]:
                    continue

                alignment = group_info.get("Alignment", None)
                tc_loc = group_info.get("Robot text loc", None)
                group_weight = group_info.get("Group weight", None)
                robot_text_box = group_info.get("Robot text box", None)

                if robot_text_box:
                    a1, b1, a2, b2 = robot_text_box
                    pdf_x1, pdf_y1, pdf_x2, pdf_y2 = self.to_pdf_rect(a1, b1, a2, b2, image_width, image_height, pdf_w, pdf_h)

                    if group_weight is not None:
                        label_left = f"{group_name}->({int(group_weight)})"
                        label_right = f"({int(group_weight)})->{group_name}"
                    else:
                        label_left = group_name
                        label_right = group_name

                    has_rc = "rc" in label_left.lower()
                    has_rbt = "rbt" in label_left.lower()

                    if has_rc:
                        box_color = colors.lightgreen
                    elif has_rbt:
                        box_color = colors.skyblue
                    else:
                        box_color = colors.lightgrey

                    c.setFillColor(box_color)
                    c.setStrokeColor(colors.black)
                    c.setLineWidth(0.5 * scale_factor)
                    c.rect(pdf_x1, pdf_y1, pdf_x2 - pdf_x1, pdf_y2 - pdf_y1, fill=1, stroke=1)

                    if tc_loc:
                        tx, ty = tc_loc
                        pdf_tx, pdf_ty = self.to_pdf_coords(tx, ty, image_width, image_height, pdf_w, pdf_h)
                        c.setFont("Helvetica-Bold", 12 * scale_factor)
                        c.setFillColorRGB(0, 0, 0)
                        
                        if alignment == 'right':
                            c.drawString(pdf_tx - 40 * scale_factor, pdf_ty, label_right)
                        else:
                            c.drawString(pdf_tx, pdf_ty, label_left)

                for row_name, row_info in group_info.items():
                    if not isinstance(row_info, dict):
                        continue
                    row_bbox = row_info.get("Row bbox", None)
                    if row_bbox:
                        x1, y1, x2, y2 = row_bbox
                        pdf_x1, pdf_y1, pdf_x2, pdf_y2 = self.to_pdf_rect(x1, y1, x2, y2, image_width, image_height, pdf_w, pdf_h)

                        c.setStrokeColor(colors.blue)
                        c.setLineWidth(0.5 * scale_factor)
                        c.rect(pdf_x1, pdf_y1, pdf_x2 - pdf_x1, pdf_y2 - pdf_y1, fill=0, stroke=1)

                        row_text = f"{row_name}, {row_info.get('Box count', '')}, {group_name}"
                        c.setFont("Helvetica", 12 * scale_factor)
                        c.setFillColor(colors.black)
                        c.drawString(pdf_x1 - (100 * scale_factor), ((pdf_y2 - pdf_y1)/2) + pdf_y1, row_text)
            c.rotate(270)
            c.showPage()
            c.save()
    
            packet.seek(0)

            reader = PdfReader(pdf_path)
            writer = PdfWriter()

            overlay_reader = PdfReader(packet)
                
            overlay_page = overlay_reader.pages[0]
            
            print(f"overlay_page.rotation: {overlay_page.rotation}") 

            for page in reader.pages: 
                writer.add_page(page)

            writer.add_page(overlay_page)
            
            if not output_path:
                os.makedirs(os.path.join(self.app_dir, 'Output'), exist_ok=True)
                output_path = os.path.join(self.app_dir, 'Output', f"{image_name}-{page_id}.pdf")

            with open(output_path, "wb") as f_out:
                writer.write(f_out)

            print(f"PDF saved at {output_path}")
            return output_path

        except Exception as e:
            print(f"Error processing PDF: {e}")
            return None"""

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=200
)

result = text_splitter.split_text(text)

print(result[0])