#!/usr/bin/env python3
"""참조 문서 생성 스크립트 - 레벨별 불릿 스타일 설정"""

from docx import Document
from docx.oxml.shared import OxmlElement
from docx.oxml.ns import qn

def create_custom_numbering(doc):
    """커스텀 번호 매기기 스타일 생성 (레벨별 불릿 차별화)"""

    # numbering.xml에 접근
    numbering_part = doc.part.numbering_part
    if numbering_part is None:
        # numbering part가 없으면 생성할 수 없음
        print("⚠️ Numbering part가 없습니다.")
        return

    # 기존 abstractNum 찾기
    abstract_nums = numbering_part.element.findall(qn('w:abstractNum'))

    for abstract_num in abstract_nums:
        # 각 레벨별 설정
        for lvl_element in abstract_num.findall(qn('w:lvl')):
            ilvl = lvl_element.get(qn('w:ilvl'))

            # lvlText 요소 찾기
            lvl_text = lvl_element.find(qn('w:lvlText'))

            if ilvl == '0':
                # 레벨 1: 점 •
                if lvl_text is not None:
                    lvl_text.set(qn('w:val'), '●')
                print("✅ 레벨 1 스타일: ●")
            elif ilvl == '1':
                # 레벨 2: 대시 -
                if lvl_text is not None:
                    lvl_text.set(qn('w:val'), '–')
                print("✅ 레벨 2 스타일: –")
            elif ilvl == '2':
                # 레벨 3: 사각형 ▪
                if lvl_text is not None:
                    lvl_text.set(qn('w:val'), '▪')
                print("✅ 레벨 3 스타일: ▪")

def main():
    """참조 문서 생성"""

    # 기본 문서 열기
    doc = Document('data/reference.docx')

    print("📝 참조 문서 불릿 스타일 수정 중...")

    # 커스텀 번호 매기기 적용
    create_custom_numbering(doc)

    # 저장
    output_path = 'data/reference_custom.docx'
    doc.save(output_path)

    print(f"✅ 참조 문서 생성 완료: {output_path}")
    print("\n이제 이 문서를 Pandoc의 --reference-doc으로 사용하면")
    print("레벨별로 다른 불릿 스타일이 적용됩니다.")

if __name__ == "__main__":
    main()
