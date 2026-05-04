import xml.etree.ElementTree as ET

def get_headers():
    context = ET.iterparse('data/geco/L1_unzip/xl/worksheets/sheet1.xml', events=('end',))
    headers = []
    for event, elem in context:
        if elem.tag.endswith('row') and elem.attrib.get('r') == '1':
            for c in elem:
                v = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                if v is not None:
                    headers.append(v.text)
            break
        elem.clear()
    print("Headers:", headers)

if __name__ == "__main__":
    get_headers()