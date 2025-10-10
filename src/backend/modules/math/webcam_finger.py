import cv2
import mediapipe as mp
from typing import Any
from types import ModuleType
from numpy import dtype, integer, floating, ndarray
from src.backend.utils import HDPrint

TOP_FINGERS: list[int] = [8, 12, 16, 20]
MIDDLE_FINGERS: list[int] = [6, 10, 14, 18]


def rgb_to_bgr(*color: int) -> tuple[int, int, int]:
    r"""
    Converts an RGB color tuple to BGR format.

    Parameters:
        *color (int): Three integers representing a color in RGB format (R, G, B).

    Returns:
        tuple: A tuple representing the same color in BGR format (B, G, R).

    Notes:
        - Useful for OpenCV functions, which use BGR color ordering instead of RGB.
        - Expects exactly three integer values; passing more or fewer will raise a ValueError.
    """

    r, g, b = color
    return (b, g, r)


class Hands:
    r"""
    Class for hand detection and tracking using MediaPipe.

    This class initializes the MediaPipe Hands module, setting detection and tracking
    parameters, as well as colors for points and connections drawn on the image.
    """

    def __init__(
        self,
        mode: bool = False,
        max_hands: int = 2,
        confidence_detect: float = 0.7,
        confidence_trace: float = 0.7,
        color_points: tuple[int, int, int] = (0, 0, 255),
        color_connections: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        """
        Initializes the hand detection class using MediaPipe with configurable parameters for detection and drawing.

        This constructor sets up the MediaPipe Hands module, configures detection and tracking confidence,
        maximum number of hands to detect, and drawing specifications for landmarks and connections.

        Parameters:
            mode (bool, optional): Operating mode for MediaPipe Hands. False for static image detection,
                True for continuous tracking. Default is False.
            max_hands (int, optional): Maximum number of hands to detect. Default is 2.
            confidence_detect (float, optional): Minimum confidence threshold for detecting hands. Default is 0.7.
            confidence_trace (float, optional): Minimum confidence threshold for tracking hand landmarks. Default is 0.7.
            color_points (tuple[int, int, int], optional): RGB color for hand landmark points. Default is (0, 0, 255).
            color_connections (tuple[int, int, int], optional): RGB color for connections between hand landmarks. Default is (255, 255, 255).

        Attributes:
            _mp_hands: MediaPipe Hands module.
            _hands: Configured MediaPipe Hands instance.
            _result: Stores the latest detection result (hand landmarks and handedness).
            _drawing: MediaPipe drawing utility module.
            _drawing_points: Drawing configuration for landmark points.
            _drawing_connections: Drawing configuration for connections between landmarks.
        """

        self.mode: bool = mode
        self.max_hands: int = max_hands
        self.confidence_detect: float = confidence_detect
        self.confidence_trace: float = confidence_trace
        self.color_points: tuple[int, int, int] = color_points
        self.color_connections: tuple[int, int, int] = color_connections
        self._mp_hands: ModuleType = mp.solutions.hands
        self._hands: mp.solutions.hands.Hands = self._mp_hands.Hands(
            self.mode, self.max_hands, 1, self.confidence_detect, self.confidence_trace
        )
        self._result: Any = None
        self._drawing: ModuleType = mp.solutions.drawing_utils
        self._drawing_points: mp.solutions.drawing_utils.DrawingSpec = (
            self._drawing.DrawingSpec(color=self.color_points)
        )
        self._drawing_connections: mp.solutions.drawing_utils.DrawingSpec = (
            self._drawing.DrawingSpec(color=self.color_connections)
        )

    @staticmethod
    def _count_fingers(hand_landmarks: Any, handedness_index: int) -> int:
        r"""
        Counts the number of fingers that are extended for a given hand.

        This method analyzes the landmarks of a hand and determines which fingers
        are extended based on the relative positions of finger joints. It also
        accounts for whether the hand is left or right.

        Parameters
        ----------
        hand_landmarks : Any
            MediaPipe hand landmarks object containing normalized landmark coordinates.
        handedness_index : int
            Indicator of the hand side: 0 for left hand, 1 for right hand.

        Returns
        -------
        int
            The number of fingers that are extended (0 to 5).

        Notes
        -----
        - The method compares the y-coordinates of top and middle finger joints for
        index, middle, ring, and pinky fingers.
        - For the thumb, it compares the x-coordinates of the thumb tip and
        its preceding joint, taking handedness into account.
        - This function is intended for internal use only (prefixed with an underscore).
        """

        hand_lm: Any = hand_landmarks.landmark
        count_fg: int = 0

        for tf, mf in zip(TOP_FINGERS, MIDDLE_FINGERS):
            count_fg += 1 if hand_lm[tf].y < hand_lm[mf].y else 0

        if handedness_index:
            count_fg += 1 if hand_lm[4].x < hand_lm[3].x else 0
        else:
            count_fg += 1 if hand_lm[4].x > hand_lm[3].x else 0

        return count_fg

    def search_hands(
        self,
        frame: cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]],
        drawing: bool = True,
    ) -> cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]]:
        r"""
        Detects hands in a given frame using MediaPipe and optionally draws landmarks and connections.

        This method converts the input frame to RGB, processes it with MediaPipe Hands to detect
        hand landmarks, and stores the detection results internally. If the `drawing` parameter
        is True, it draws the detected hand landmarks and their connections on the original frame
        using the configured drawing specifications.

        Parameters:
            frame (cv2.Mat | ndarray): The input image or video frame in BGR format.
            drawing (bool, optional): If True, draws landmarks and hand connections on the frame.
                Default is True.

        Returns:
            cv2.Mat | ndarray: The original frame with optional drawings of detected hand landmarks
            and connections.

        Notes:
            - Detected hand landmarks and handedness information are stored internally in `self._result`.
            - Uses MediaPipe's predefined HAND_CONNECTIONS to draw lines between landmarks.
            - The drawing colors and styles are defined by `_drawing_points` and `_drawing_connections`.
        """

        frame_rgb: cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        self._result = self._hands.process(frame_rgb)

        if self._result.multi_hand_landmarks and self._result.multi_handedness:
            for hand_landmarks in self._result.multi_hand_landmarks:
                if drawing:
                    self._drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self._mp_hands.HAND_CONNECTIONS,
                        self._drawing_points,
                        self._drawing_connections,
                    )
        return frame

    def search_points(
        self,
        frame: cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]],
        drawing: bool = True,
        color_radius: tuple[int, int, int] = rgb_to_bgr(0, 0, 0),
        color_landmark_id: tuple[int, int, int] = rgb_to_bgr(0, 0, 255),
        radius: int = 7,
        landmark_id: bool = False,
    ) -> tuple[
        cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]], list[list[int]]
    ]:
        r"""
        Extracts hand landmark coordinates from a frame and optionally draws them with visual markers.

        This method processes the detected hand landmarks stored in `self._result`, converts their
        normalized coordinates to pixel values relative to the frame dimensions, and appends them
        to a list. Optionally, it draws circles at each landmark and can display the landmark IDs.

        Parameters:
            frame (cv2.Mat | ndarray): The input image/frame in which hands have been detected.
            drawing (bool, optional): If True, draws circles at each landmark position. Default is True.
            color_radius (tuple[int, int, int], optional): RGB color of the landmark circles. Default is black (0, 0, 0) via `rgb_to_bgr`.
            color_landmark_id (tuple[int, int, int], optional): RGB color of landmark IDs if displayed. Default is red (0, 0, 255) via `rgb_to_bgr`.
            radius (int, optional): Radius of the landmark circles. Default is 7.
            landmark_id (bool, optional): If True, displays the landmark ID next to each point. Default is False.

        Returns:
            tuple:
                - cv2.Mat | ndarray: The frame with optional drawings of landmarks and IDs.
                - list[list[int]]: List of landmark coordinates in the format [id, x, y].

        Notes:
            - Coordinates are scaled to match the dimensions of the input frame.
            - Requires `self._result` to contain valid hand detection results from MediaPipe.
            - Disabling drawing or landmark IDs can improve performance in real-time processing.
            - Color values are converted from RGB to BGR using `rgb_to_bgr`.
        """

        coordinates: list = []

        if self._result.multi_hand_landmarks and self._result.multi_handedness:
            for hand_landmarks, handedness in zip(
                self._result.multi_hand_landmarks, self._result.multi_handedness
            ):
                for id, points in enumerate(hand_landmarks.landmark):
                    h, w, _ = frame.shape
                    cx, cy = int(points.x * w), int(points.y * h)
                    coordinates.append([id, cx, cy])
                    if drawing:
                        cv2.circle(frame, (cx, cy), radius, color_radius, cv2.FILLED)
                        if landmark_id:
                            if handedness.classification[0].index == 1:
                                cv2.putText(
                                    frame,
                                    str(id),
                                    (cx, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color_landmark_id,
                                    2,
                                )
                            else:
                                cv2.putText(
                                    frame,
                                    str(id),
                                    (cx + 5, cy + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color_landmark_id,
                                    2,
                                )
        return frame, coordinates

    @property
    def count_left_point(self) -> int:
        r"""
        Returns the number of fingers extended on the left hand.

        This property checks the detected hands in `self.result` and counts the
        fingers of the hand classified as left (index 0) using the `_count_fingers` method.

        Returns
        -------
        int
            Number of fingers extended on the left hand (0 to 5).

        Notes
        -----
        - If no left hand is detected, returns 0.
        - Relies on `self.result.multi_hand_landmarks` and `self.result.multi_handedness`.
        """

        count_left: int = 0

        if self._result.multi_hand_landmarks and self._result.multi_handedness:
            for hand_landmarks, handedness in zip(
                self._result.multi_hand_landmarks, self._result.multi_handedness
            ):
                if handedness.classification[0].index == 0:
                    count_left = Hands._count_fingers(hand_landmarks, 0)
        return count_left

    @property
    def count_right_point(self) -> int:
        r"""
        Returns the number of fingers extended on the right hand.

        This property checks the detected hands in `self.result` and counts the
        fingers of the hand classified as right (index 1) using the `_count_fingers` method.

        Returns
        -------
        int
            Number of fingers extended on the right hand (0 to 5).

        Notes
        -----
        - If no right hand is detected, returns 0.
        - Relies on `self.result.multi_hand_landmarks` and `self.result.multi_handedness`.
        """

        count_right: int = 0
        if self._result.multi_hand_landmarks and self._result.multi_handedness:
            for hand_landmarks, handedness in zip(
                self._result.multi_hand_landmarks, self._result.multi_handedness
            ):
                if handedness.classification[0].index == 1:
                    count_right = Hands._count_fingers(hand_landmarks, 1)
        return count_right


class WebCamActivate(Hands):
    r"""
    Initializes the WebCamActivate class, extending the Hands class, to manage webcam capture
    and hand detection with configurable display and drawing options.

    This constructor sets up the webcam, window properties, color schemes for drawing landmarks
    and connections, debug options, and hand detection parameters. It also initializes the
    parent Hands class with the configured parameters and ensures the webcam is accessible.

    Attributes:
        # Immutable attributes
        _cap (cv2.VideoCapture): Video capture object for accessing the webcam.
        _ret (bool): Flag indicating if the last frame was successfully read.
        _frame (cv2.Mat | ndarray): Stores the last captured frame.
        _print_points (list[list[int]] | None): Stores coordinates of detected landmarks for debugging.

        # Window attributes
        window_name (str): Name of the display window. Default is "Count Fingers".
        window_width (int): Width of the display window in pixels. Default is 1280.
        window_height (int): Height of the display window in pixels. Default is 720.
        window_width_frame (int): Width of the captured frame in pixels. Default is 1920.
        window_height_frame (int): Height of the captured frame in pixels. Default is 1080.
        fps (int): Frames per second for capturing video. Default is 30.
        gain (int): Webcam gain setting. Default is 0.

        # Color attributes
        color_points (tuple[int, int, int]): RGB color for hand landmark points.
        color_connections (tuple[int, int, int]): RGB color for landmark connections.
        color_radius (tuple[int, int, int]): RGB color for drawing points with radius.
        color_landmark_id (tuple[int, int, int]): RGB color for landmark IDs.
        color_rectangle (tuple[int, int, int]): RGB color for rectangles (if used).
        color_left (tuple[int, int, int]): RGB color for left hand display.
        color_right (tuple[int, int, int]): RGB color for right hand display.
        color_total (tuple[int, int, int]): RGB color for total points display.

        # Print/debug attributes
        points_debug (bool): If True, prints debug information about points.

        # Hand detection attributes
        mode (bool): MediaPipe Hands mode (False for static, True for tracking). Default is False.
        max_hands (int): Maximum number of hands to detect. Default is 2.
        confidence_detect (float): Detection confidence threshold. Default is 0.7.
        confidence_trace (float): Tracking confidence threshold. Default is 0.7.
        drawing_hands (bool): Whether to draw detected hands. Default is True.
        drawing_points (bool): Whether to draw individual landmark points. Default is True.
        landmark_id (bool): Whether to display landmark IDs. Default is True.
        size_radius (int): Radius of the drawn points. Default is 7.

    Raises:
        IOError: If the webcam cannot be accessed.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes the WebCamActivate instance to capture video from a webcam and manage hand detection.

        This initializer sets up:
        - Webcam access via OpenCV.
        - Display window properties (name, size, FPS, gain).
        - Color schemes for landmarks, connections, and other visual elements.
        - Debug options for printing points.
        - Hand detection parameters for MediaPipe (mode, max_hands, confidence, drawing options).
        - Initializes the parent Hands class with the configured parameters.

        No parameters are passed explicitly to __init__; all settings are configured internally.

        Raises:
            IOError: If the webcam cannot be accessed.
        """

        # ----------------------- #
        # * immutable attributes
        # ----------------------- #
        self._cap: cv2.VideoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._ret: bool = False
        self._frame: cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None
        self._print_points: list[list[int]] | None = None

        # ----------------------- #
        # * window attributes
        # ----------------------- #
        self.window_name: str = "Count Fingers"
        self.window_width: int = 1280
        self.window_height: int = 720
        self.window_width_frame: int = 1920
        self.window_height_frame: int = 1080
        self.fps: int = 30
        self.gain: int = 0

        # ----------------------- #
        # * color attributes
        # ----------------------- #
        self.color_points: tuple[int, int, int] = rgb_to_bgr(255, 0, 0)
        self.color_connections: tuple[int, int, int] = rgb_to_bgr(0, 0, 0)
        self.color_radius: tuple[int, int, int] = rgb_to_bgr(0, 255, 60)
        self.color_landmark_id: tuple[int, int, int] = rgb_to_bgr(255, 251, 0)
        self.color_rectangle: tuple[int, int, int] = rgb_to_bgr(0, 0, 0)
        self.color_left: tuple[int, int, int] = rgb_to_bgr(0, 255, 60)
        self.color_right: tuple[int, int, int] = rgb_to_bgr(0, 255, 60)
        self.color_total: tuple[int, int, int] = rgb_to_bgr(0, 255, 60)

        # ----------------------- #
        # * print attributes
        # ----------------------- #
        self.points_debug: bool = True

        # ----------------------- #
        # * hands init class attributes
        # ----------------------- #
        self.mode: bool = False
        self.max_hands: int = 2
        self.confidence_detect: float = 0.7
        self.confidence_trace: float = 0.7
        self.drawing_hands: bool = True
        self.drawing_points: bool = True
        self.landmark_id: bool = True
        self.size_radius: int = 7

        super().__init__(
            mode=self.mode,
            max_hands=self.max_hands,
            confidence_detect=self.confidence_detect,
            confidence_trace=self.confidence_trace,
            color_points=self.color_points,
            color_connections=self.color_connections,
        )

        if not self._cap.isOpened():
            raise IOError("Unable to access webcam.")

    @property
    def _config_window(self) -> None:
        r"""
        Configures the OpenCV display window.

        - Creates a resizable window with the name specified in `self.window_name`.
        - Sets the window dimensions to `self.window_width` x `self.window_height`.

        Notes
        -----
        - Should be called before displaying any frames with `cv2.imshow`.
        """

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)

    @property
    def _put_text(
        self,
    ) -> None:
        r"""
        Draws finger count information on the current frame.

        - Draws a background rectangle at the top-left corner.
        - Displays left-hand, right-hand, and total finger counts.
        - Uses configurable colors:
            - `color_rectangle`: background rectangle
            - `color_left`: left-hand count text
            - `color_right`: right-hand count text
            - `color_total`: total finger count text

        Notes
        -----
        - Relies on `self._frame` for drawing.
        - Uses `self.count_left_point` and `self.count_right_point` for counts.
        - Call after updating the frame with hand detection results.
        """

        total_count: int = self.count_left_point + self.count_right_point

        cv2.rectangle(self._frame, (5, 5), (230, 100), self.color_rectangle, -1)

        cv2.putText(
            self._frame,
            f"Left Hand: {self.count_left_point}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.color_left,
            2,
        )
        cv2.putText(
            self._frame,
            f"Right Hand: {self.count_right_point}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.color_right,
            2,
        )
        cv2.putText(
            self._frame,
            f"Total Fingers: {total_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.color_total,
            2,
        )

    @property
    def _config_webcam(self) -> None:
        r"""
        Configures webcam capture properties.

        - Sets the frame width and height (`self.window_width_frame`, `self.window_height_frame`).
        - Sets the capture FPS (`self.fps`).
        - Sets the webcam gain (`self.gain`).

        Notes
        -----
        - Must be called before starting the main capture loop to ensure correct settings.
        """

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window_width_frame)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window_height_frame)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_GAIN, self.gain)

    def mainloop(self) -> None:
        r"""
        Runs the main loop for capturing webcam frames, detecting hands, and displaying the output.

        This method continuously captures frames from the webcam, flips them horizontally for
        mirror view, detects hands using MediaPipe, extracts landmark points, optionally draws
        landmarks and IDs, prints debug information if enabled, and displays the processed frame
        in a window. The loop continues until the ESC key is pressed or the window is closed.

        No parameters are required.

        Raises:
            RuntimeError: If a frame cannot be captured from the webcam.

        Notes:
            - Calls `_config_window` and `_config_webcam` internally to set up the display and capture.
            - Uses `search_hands` to detect hand landmarks and `search_points` to get coordinates.
            - Uses `self.points_debug` to optionally print landmark coordinates.
            - The window name and visual settings are configured via instance attributes.
            - Automatically releases the webcam and destroys all OpenCV windows when exiting.
        """

        self._config_window

        self._config_webcam

        while True:

            self._ret, self._frame = self._cap.read()
            if not self._ret:
                raise RuntimeError("Failed to capture frame.")

            self._frame = cv2.flip(self._frame, 1)

            self._frame = self.search_hands(
                frame=self._frame, drawing=self.drawing_hands
            )

            self._frame, self._print_points = self.search_points(
                frame=self._frame,
                drawing=self.drawing_points,
                color_radius=self.color_radius,
                color_landmark_id=self.color_landmark_id,
                radius=self.size_radius,
                landmark_id=self.landmark_id,
            )

            if self.points_debug:
                HDPrint(self._print_points).print()

            self._put_text

            cv2.imshow(self.window_name, self._frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        self._cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    webcam = WebCamActivate()
    webcam.mainloop()

# python -W ignore -m src.backend.modules.math.webcam_finger
