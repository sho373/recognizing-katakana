from xml.etree import ElementTree
import os
import shutil

label_xml_dir = "synth_text_images"
out_dir = "txt"
img_dir = "images"

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(os.path.join(label_xml_dir,img_dir)):
    os.mkdir(os.path.join(label_xml_dir,img_dir))

for filename in os.listdir(label_xml_dir):
    if filename.endswith(".jpg"):
        shutil.copy(os.path.join(label_xml_dir,file_name), os.path.join(label_xml_dir,img_dir))
        
    if filename.endswith(".xml"):
       
        full_file = os.path.abspath(os.path.join(label_xml_dir,filename))
        dom = ElementTree.parse(full_file)
        objs = dom.findall('object')
        
        for o in objs:
            tl,tr,br,bl = [],[],[],[]
            words = o.find('extra').text
            tl.append(o.find('bndbox/xmin').text)
            tl.append(o.find('bndbox/ymin').text)
            br.append(o.find('bndbox/xmax').text)
            br.append(o.find('bndbox/ymax').text)
            tr.append(br[0])
            tr.append(tl[1])
            bl.append(tl[0])
            bl.append(br[1])

            with open('{}/gt_{}.txt'.format(out_dir,filename.split(".xml")[0]),'a',encoding='utf8') as f:
                        f.write("{},{},{},{},{},{},{},{},{}\n".format(tl[0],tl[1],tr[0],tr[1],br[0],br[1],bl[0],bl[1],words))
        print(filename)