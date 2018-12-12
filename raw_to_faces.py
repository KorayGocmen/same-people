import os, uuid, cv2
import face_detector

raw_folder = "raw"
faces_folder = "faces"
blurry_folder = "blurry"
face_count = 0

def get_subdirectories(a_dir):
  return [name for name in os.listdir(a_dir) 
    if os.path.isdir(os.path.join(a_dir, name))]

def is_blurry(image_path, threshold=90):
  image = cv2.imread(image_path)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

if __name__ == "__main__":
  
  detector = face_detector.FaceDetector(image_size=128)

  if not os.path.exists(faces_folder):
    os.makedirs(faces_folder)
  if not os.path.exists(blurry_folder):
    os.makedirs(blurry_folder)

  subdirs = get_subdirectories(raw_folder)

  for subdir in subdirs:
    faces_dir = faces_folder + "/" + subdir
    if not os.path.exists(faces_dir):
      os.makedirs(faces_dir)
    
    raw_dir = raw_folder + "/" + subdir
    files = os.listdir(raw_dir)
    
    for file in files:
      raw_path = raw_dir + "/" + file
      faces_path = faces_dir + "/" + file

      succes = detector.detect(raw_path, faces_path)
      if succes:
        # if is_blurry(faces_path):
        #   blurry_path = blurry_folder + "/" + str(uuid.uuid4()) + ".jpg"
        #   os.rename(faces_path, blurry_path)
        # else:
        #   face_count += 1
        face_count += 1

  print("DONE", face_count)