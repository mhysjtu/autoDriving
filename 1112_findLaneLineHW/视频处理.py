import cv2
# cv2.cv as cv
cap = cv2.VideoCapture('video.mp4')
# in opencv3.0, CAP_PROP_FPS has no prefix of CV_
nbframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
duration = (nbframes * fps) / 1000
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Num. Frames = ', nbframes)
print('Frame Rate = ', fps, 'fps')
print('Duration = ', duration, 'sec')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, fps, (w,h))

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret == True:
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		out.write(frame)
		cv2.imshow('frame',gray)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
cap.release()
out.release()
cv2.destroyAllWindows()