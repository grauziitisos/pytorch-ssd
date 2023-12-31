#import xmltodict
import xml.etree.ElementTree as ET
from pathlib import Path

directory = 'F:\\_o\\Users\\w\\3D Objects\\qfgaohao_py-ssd\\pytorch-ssd\\pas\\train\\'
#dataset_file = r'000c6c92c80213ff.xml' # The path to the XML file
files = Path(directory).glob('*.xml')
for dataset_file in files:
    xml_tree = ET.parse(dataset_file) # Parse the XML file
    root = xml_tree.getroot() # Find the root element

    assert root.tag == 'annotation' or root.attrib['verified'] == 'yes', "PASCAL VOC does not contain a root element" # Check if the root element is "annotation"
    #assert len(root.findtext('folder')) > 0, "XML file does not contain a 'folder' element"
    assert len(root.findtext('filename')) > 0, "XML file does not contain a 'filename'"
    #assert len(root.findtext('path')) > 0, "XML file does not contain 'path' element"
    #assert len(root.find('source')) == 1 and len(root.find('source').findtext('database')) > 0, "XML file does not contain 'source' element with a 'database'"
    #assert len(root.find('size')) == 3, "XML file doesn not contain 'size' element"
    assert root.find('size').find('width').text and root.find('size').find('height').text , "XML file does not contain either 'width', 'height'" 
    #assert root.find('size').find('width').text and root.find('size').find('height').text and root.find('size').find('depth').text, "XML file does not contain either 'width', 'height', or 'depth' element"
    #assert root.find('segmented').text == '0' or len(root.find('segmented')) > 0, "'segmented' element is neither 0 or a list"
    assert len(root.findall('object')) > 0, "XML file contains no 'object' element" # Check if the root contains zero or more 'objects'
    width = int(root.find('size').find('width').text )
    height = int(root.find('size').find('height').text)
    required_objects = ['name', 'pose', 'truncated', 'difficult', 'bndbox'] # All possible meta-data about an object
    for obj in root.findall('object'):
    #  assert len(obj.findtext(required_objects[0])) > 0, "Object does not contain a parameter 'name'"
    #  assert len(obj.findtext(required_objects[1])) > 0, "Object does not contain a parameter 'pose'"
    #  assert int(obj.findtext(required_objects[2])) in [0, 1], "Object does not contain a parameter 'truncated'"
    #  assert int(obj.findtext(required_objects[3])) in [0, 1], "Object does not contain a parameter 'difficult'"
      assert len(obj.findall(required_objects[4])) > 0, "Object does not contain a parameter 'bndbox'"
      for bbox in obj.findall(required_objects[4]):
        assert int(float(bbox.findtext('xmin'))) is not None, "'xmin' value for the bounding box is missing "
        assert float(bbox.findtext('ymin')) is not None, "'ymin' value for the bounding box is missing "
        assert float(bbox.findtext('xmax'))  is not None, "'xmax' value for the bounding box is missing "
        assert float(bbox.findtext('ymax'))  is not None, "'ymax' value for the bounding box is missing "

        assert int(float(bbox.findtext('xmin'))) <= width, "'xmin' value for the bounding box is > width "
        assert float(bbox.findtext('ymin')) <= height, "'ymin' value for the bounding box is > height "
        assert float(bbox.findtext('xmax')) <= width, "'xmax' value for the bounding box is > width "
        assert float(bbox.findtext('ymax')) <= height, "'ymax' value for the bounding box is > height "


    print(str(dataset_file) + ' is OK')
          #The dataset format is PASCAL VOC!')