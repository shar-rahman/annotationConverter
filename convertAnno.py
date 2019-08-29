import os
import csv
import xml.etree.ElementTree as ET
import xml.etree.cElementTree as ETc
from PIL import Image
import sys
import string
import cv2
import shutil

# def classes here ( IN SSD )
# in ssd, background is 0
# dont define it.
CLASS_MAPPING = {
    '1': 'head',
    '2': 'body',
    '3': 'whiteboard',
}

def printUsage():
    print("Usage: python %s <src_anno_type> <dest_anno_type> <src_dir> optional:<image_dir>" % sys.argv[0])
    print("Supported src annotation types:")
    print("- PASCALVoc\n- YOLO\n- SSD\n")
    print("Supported dest annotation types:")
    print("- YOLO\n- SSD\n- PASCALVoc")
    print("Must def. image dir !!!")
    exit(-1)

if len(sys.argv) != 5:
    printUsage()

if sys.argv[1].upper() != "PASCALVOC" and sys.argv[1].upper() != "YOLO" and sys.argv[1].upper() != "SSD":
    print("Invalid src anno type")
    printUsage()
else: sourceAnno = sys.argv[1].upper()

if not sys.argv[2].upper() == "YOLO" and not sys.argv[2].upper() == "SSD" and not sys.argv[2].upper() == "PASCALVOC":
    print("Invalid dest anno type")
    printUsage()
else: destAnno = sys.argv[2].upper()

if not os.path.exists(sys.argv[3]):
    print("src dir does not exist")
    printUsage()
else: sourceDir = sys.argv[3] + "/"

if sourceAnno == "YOLO" or destAnno == "YOLO" or sourceAnno == "PASCALVOC" or destAnno == "PASCALVOC":
    if len(sys.argv) != 5: printUsage()
    if not os.path.exists(sys.argv[4]):
        print("image dir does not exist - you need it")
        printUsage()
    else: imageDir = sys.argv[4] + "/"

if os.path.exists(destAnno):
    print("Detected files in destination directory: %s" % destAnno)
    raw = input("Would you like to clear this directory to prevent overwriting [y/n]: ")
    if raw.upper() == "Y": shutil.rmtree(destAnno+"/")

if not os.path.exists(destAnno):
    os.makedirs(destAnno)
destDir = destAnno + "/"

if not os.path.exists("TEMP/"):
    os.makedirs("TEMP/")

# TODO: fix hardcoding classes lmao
def convertPASCAL(sourceDir):
    if not os.path.exists("TEMP/"): os.makedirs("TEMP/")
    for filename in os.listdir(sourceDir):
        tree = ET.parse(sourceDir + filename)
        root = tree.getroot()
        for item in root.findall('object'):
            for child in item:
                if child.tag == 'name':
                    classTitle = child.text.encode('utf8')
                    if str(classTitle) == "b'head'":
                        _class = 1
                    elif str(classTitle) == "b'body'":
                        _class = 2
                    elif str(classTitle) == "b'whiteboard'" or str(classTitle) == "b'wtd'":
                        _class = 3
                    else: _class = 4
                if child.tag == 'bndbox':
                    minX = int(child[0].text.encode('utf8'))
                    minY = int(child[1].text.encode('utf8'))
                    maxX = int(child[2].text.encode('utf8'))
                    maxY = int(child[3].text.encode('utf8'))
            print("%s %s %s %s %s" % (_class, minX, minY, maxX, maxY))
            with open("TEMP/" + filename[:-4]+".txt", "a+") as f:
                f.write("%d %d %d %d %d\n" % (_class, minX, minY, maxX, maxY))

################################################################################3
# pascal section
def create_root(file_prefix, width, height):
    root = ETc.Element("annotations")
    ETc.SubElement(root, "filename").text = "{}.png".format(file_prefix)
    ETc.SubElement(root, "folder").text = "images"
    size = ETc.SubElement(root, "size")
    ETc.SubElement(size, "width").text = str(width)
    ETc.SubElement(size, "height").text = str(height)
    ETc.SubElement(size, "depth").text = "3"
    return root

def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ETc.SubElement(root, "object")
        ETc.SubElement(obj, "name").text = voc_label[0]
        ETc.SubElement(obj, "pose").text = "Unspecified"
        ETc.SubElement(obj, "truncated").text = str(0)
        ETc.SubElement(obj, "difficult").text = str(0)
        bbox = ETc.SubElement(obj, "bndbox")
        ETc.SubElement(bbox, "xmin").text = str(voc_label[1])
        ETc.SubElement(bbox, "ymin").text = str(voc_label[2])
        ETc.SubElement(bbox, "xmax").text = str(voc_label[3])
        ETc.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root

def create_file(file_prefix, width, height, voc_labels, destDir):
    root = create_root(file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ETc.ElementTree(root)
    tree.write("{}/{}.xml".format(destDir, file_prefix))

def read_file(file_path, imageDir, sourceDir, destDir):
    file_prefix = file_path.split(".txt")[0]
    image_file_name = "{}.png".format(file_prefix)
    img = Image.open("{}/{}".format(imageDir, image_file_name))
    w, h = img.size
    with open(sourceDir + file_path, 'r') as file:
        lines = file.readlines()
        voc_labels = []
        for line in lines:
            voc = []
            line = line.strip()
            data = line.split()
            voc.append(CLASS_MAPPING.get(data[0]))
            xmin = int(data[1])
            ymin = int(data[2])
            xmax = int(data[3])
            ymax = int(data[4])
            voc.append(xmin)
            voc.append(ymin)
            voc.append(xmax)
            voc.append(ymax)
            voc_labels.append(voc)
        create_file(file_prefix, w, h, voc_labels, destDir)


def convertSSDtoPascal(sourceDir, destDir, imageDir):
    if not os.path.exists(destDir):
        os.makedirs(destDir)
    for filename in os.listdir(sourceDir):
        if filename.endswith('.txt'):
            read_file(filename, imageDir, sourceDir, destDir)
        else: print("Invalid Format: %s" % filename)
# end convert to pascal
###############################################################################

###############################################################################
# start yolo conversion
def getIdandMinMax(tokens, width, height):
    cid = int(tokens[0])
    centerX = float(tokens[1])
    centerY = float(tokens[2])
    boxWidth = float(tokens[3])
    boxHeight = float(tokens[4])

    minX = centerX - (boxWidth / 2)
    minY = centerY - (boxHeight / 2)
    maxX = centerX + (boxWidth / 2)
    maxY = centerY + (boxHeight / 2)
    
    return (cid + 1, int(minX * width), int(minY * height), int(maxX * width), int(maxY * height))

def convertYOLO(sourceDir, imageDir):
    print("Converting YOLO")
    if not os.path.exists("TEMP/"): os.makedirs("TEMP/")
    for filename in os.listdir(imageDir):
        imageObject = cv2.imread(imageDir + filename)
        textFile = filename[:-4] + ".txt"
        if not os.path.exists(sourceDir + textFile): 
            print("Annotation not found for: %s" % filename)
            continue
        if imageObject is None:
            print("Image %s can not bed read" % filename)
            continue
        h,w,_ = imageObject.shape
        newFile = "TEMP/" + textFile
        with open(sourceDir + textFile, "r") as rfd:
            with open(newFile, "a+") as wfd:
                for line in rfd:
                    tokens = line.split(" ")
                    if len(tokens) != 5: raise Exception("Invalid anno: %s" % line)
                    idMinMax = getIdandMinMax(tokens, w, h)
                    if idMinMax != None:
                        wfd.write("%d %d %d %d %d\n" % idMinMax)

def convertIdandMinMax(tokens, w, h):
    print("Trying: ", w, h)
    cid = int(tokens[0])
    boxWidth = float((int(tokens[3]) - int(tokens[1])) / int(w))
    boxHeight = float((int(tokens[4]) - int(tokens[2])) / int(h))
    centerX = float(((int(tokens[1]) + int(tokens[3])) / 2) / int(w))
    centerY = float(((int(tokens[4]) + int(tokens[2])) / 2) / int(h))
    return (cid - 1, centerX, centerY, boxWidth, boxHeight)


def convertSSDtoYOLO(sourceDir, destDir, imageDir):
    for filename in os.listdir(sourceDir):
        imageName = imageDir + filename[:-4] + ".png"
        imageObject = cv2.imread(imageDir + filename[:-4] + ".png")
        textFile = destDir + filename
        h,w,_ = imageObject.shape
        with open(sourceDir + filename, "r") as rfd:
            with open(textFile, "a+") as wfd:
                for line in rfd:
                    tokens = line.split(" ")
                    if len(tokens) != 5: raise Exception("Invalid anno: %s" % filename)
                    idMinMax = convertIdandMinMax(tokens, w, h)
                    if idMinMax is not None:
                        wfd.write("%s %s %s %s %s\n" % idMinMax)

# end yolo conversion
#########################################################################33
if os.path.exists("TEMP/"):
    shutil.rmtree("TEMP/")


# Reads in what type of annotation the user has and turns it into SSD
# Then depending on the destination, converts SSD into that.
def AnnoManager(sourceAnno, destAnno, sourceDir, destDir, imageDir):
    if sourceAnno == "PASCALVOC": convertPASCAL(sourceDir)
    if sourceAnno == "YOLO": convertYOLO(sourceDir, imageDir)
    if sourceAnno == "SSD": os.rename(sourceDir, "TEMP/")
    if destAnno == "SSD": 
        for f in os.listdir("TEMP/"):
            shutil.move("TEMP/" + f, destDir)
    if destAnno == "YOLO": convertSSDtoYOLO("TEMP/", destDir, imageDir)
    if destAnno == "PASCALVOC": convertSSDtoPascal("TEMP/", destDir, imageDir)

print("Converting %s to %s." % (sourceAnno, destAnno))
AnnoManager(sourceAnno, destAnno, sourceDir, destDir, imageDir)
print("Done converting. Annotations found in: %s directory." % destDir)
if os.path.exists("TEMP/") and sourceAnno == "SSD":
    os.rename("TEMP/", sourceDir)
else: shutil.rmtree("TEMP/")