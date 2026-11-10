from pathlib import Path
import re,json,html,xml.etree.ElementTree as ET
import markdown
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
root=Path(__file__).resolve().parents[1]
md=(root/'report.md').read_text(encoding='utf-8-sig')
# Interpret the user's title and missing heading-space as formatting only.
# Preserve all wording in the authoritative Markdown file.
display_md=re.sub(r'^##AI使用聲明：', '## AI使用聲明：\n\n', md, flags=re.M)
if not display_md.startswith('#'):
 display_md='# '+display_md
body=markdown.markdown(display_md,extensions=['tables'])
(root/'output/report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:850px;margin:40px auto;font:18px/1.8 sans-serif}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style>'+body,encoding='utf-8')
pdfmetrics.registerFont(TTFont('ZH','C:/Windows/Fonts/msjh.ttc',subfontIndex=0))
base=ParagraphStyle('base',fontName='ZH',fontSize=10.5,leading=17,spaceAfter=9,wordWrap='CJK',allowWidows=0,allowOrphans=0)
heading=ParagraphStyle('heading',parent=base,fontSize=13,leading=21,spaceBefore=12,keepWithNext=True)
title=ParagraphStyle('title',parent=heading,fontSize=18,leading=26,spaceAfter=16)
flows=[]
def p(t,style=base):return Paragraph(html.escape(t),style)
for el in ET.fromstring('<root>'+body+'</root>'):
 if el.tag=='table':
  rows=[[p(''.join(c.itertext())) for c in r] for r in el.findall('.//tr')]
  table=Table(rows,colWidths=[(A4[0]-100)/len(rows[0])]*len(rows[0]),repeatRows=1)
  table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5efeb')),('GRID',(0,0),(-1,-1),.4,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));flows.extend([table,Spacer(1,10)])
 else:flows.append(p(''.join(el.itertext()),title if el.tag=='h1' else heading if el.tag=='h2' else base))
def footer(c,d):
 c.setFont('ZH',8);c.drawRightString(A4[0]-50,25,str(d.page))
SimpleDocTemplate(str(root/'report.pdf'),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=45,title='辨識冒犯，還是辨識冒犯詞？',author='').build(flows,onFirstPage=footer,onLaterPages=footer)
t=md.split('## 參考資料')[0];counts={'han_characters_including_title_tables':len(re.findall(r'[\u4e00-\u9fff]',t)),'status':'user revision with requested anti-bias difficulty paragraph','note':'References and AI declaration excluded; Moodle submission is separate.'}
(root/'output/report_check.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2),encoding='utf-8');print(counts)
