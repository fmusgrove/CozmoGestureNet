from HandDetector.utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from threading import Thread
from multiprocessing import Queue, Pool
import time
from HandDetector.utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import cozmo
import numpy

from HandDetector import image_grab

frame_processed = 0
score_thresh = 0.15


# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue


def worker(input_queue, output_queue, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    while True:
        # print("> ===== in worker loop, frame ", frame_processed)
        frame = input_queue.get()
        if frame is not None:
            # Actual detection. Variable boxes contains the bounding box coordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found at least one hand (within your score threshold)

            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # draw bounding boxes
            roi_bounds = detector_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            # add frame annotated with bounding box to queue
            output_queue.put((frame, roi_bounds))
            frame_processed += 1
        else:
            output_queue.put((frame, None))
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=320,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=240,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = (320, 240)
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    # spin up workers to parallelize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Isolated Hand', cv2.WINDOW_NORMAL)

    cozmo_thread = Thread(target=image_grab.run_cozmo_photostream, daemon=True).start()

    print('Waiting for image')
    x = 0
    while True:
        # Wait for a new image to be available from Cozmo
        # wait_for_image = image_grab.photo_stream.image_available.wait()

        # Grab the image and clear the event
        frame = image_grab.photo_stream.latest_image
        # image_grab.photo_stream.image_available.clear()

        if frame is not None:
            # Convert the frame from a PIL image to an OpenCV image format
            frame = numpy.array(frame)

            # Convert from RGB to BGR
            frame = frame[:, :, ::-1].copy()

            frame = cv2.flip(frame, 1)
            index += 1

            input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            output_frame, bounds = output_q.get()

            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            num_frames += 1
            fps = num_frames / elapsed_time
            # print("frame ",  index, num_frames, elapsed_time, fps)

            if output_frame is not None:
                if args.display > 0:
                    if bounds is not None:
                        if len(bounds) > 0:
                            # isolated_hand = output_frame[y1:y2, x1:x2]
                            p1, p2 = bounds[0]
                            isolated_hand = output_frame[p1[1]:p2[1], p1[0]:p2[0]]
                            cv2.imshow('Isolated Hand', isolated_hand)
                    if args.fps > 0:
                        detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                         output_frame)
                    cv2.imshow('Multi-Threaded Detection', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if num_frames == 400:
                        num_frames = 0
                        start_time = datetime.datetime.now()
                    else:
                        print("frames processed: ", index, "elapsed time: ",
                              elapsed_time, "fps: ", str(int(fps)))
            else:
                # print("video end")
                break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print('Ending FPS:', fps)
    pool.terminate()
    cv2.destroyAllWindows()
