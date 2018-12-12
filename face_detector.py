import cv2, sys, os, pathlib

class FaceDetector(object):
  FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"

  def __init__(self, image_size=32):
    self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_PATH)
    self.IMAGE_SIZE = image_size

  def get_face(self, image_path, new_image_path=None):
    try:
      img = cv2.imread(image_path)
      if (img is None):
        print("image not found " + image_path)
        return 0, None

      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = self.face_cascade.detectMultiScale(img, 1.1, 5)
      if (len(faces) == 0):
        print("no face in " + image_path)
        return False, None

      if (len(faces) > 1):
        print("multiple faces in " + image_path)
        return False, None
    
      height, width = img.shape[:2]
      img_name = pathlib.Path(image_path).name.split(".")[0]
      img_path_only = pathlib.Path(image_path).parents[0]

      i = 0
      for (x,y,w,h) in faces:
        r = max(w, h) / 2
        centerx = x + w / 2
        centery = y + h / 2
        nx = int(centerx - r)
        ny = int(centery - r)
        nr = int(r * 2)

        face_img = img[ny:ny+nr, nx:nx+nr]
        face_img_resized = cv2.resize(face_img, (self.IMAGE_SIZE, self.IMAGE_SIZE))

        i += 1
        if (new_image_path is not None):
          cv2.imwrite(str(i)+new_image_path, face_img_resized)

        return True, face_img_resized
    except Exception as e:
      print("unknown error in detector", e)
      return False, None

  def detect(self, image_path, new_image_path=None):
    detected, image = self.get_face(image_path, new_image_path)
    return detected

if __name__ == "__main__":
  
  detector = FaceDetector(image_size=256)
  succes = detector.detect("/Users/koraygocmen/Documents/github/same-people/raw/806/0.jpg", "01.jpg")
  print(succes)