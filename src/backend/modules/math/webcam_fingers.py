import cv2
import mediapipe as mp
from itertools import product
from typing import Any
from types import ModuleType
from numpy import dtype, integer, floating, ndarray
from src.backend.utils import HDPrint


def rgb_to_bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    r"""
    Convert an RGB color to BGR format.

    Args:
        r (int): Red component (0-255).
        g (int): Green component (0-255).
        b (int): Blue component (0-255).

    Returns:
        return (tuple[int, int, int]): The color represented in BGR format as (B, G, R).

    Notes:
        - Useful for OpenCV, which expects colors in BGR order instead of RGB.
    """
    return (b, g, r)


class Hands:
    r"""
    Class for hand detection and tracking using MediaPipe.

    This class initializes the MediaPipe Hands module, setting detection and tracking
    parameters, as well as colors for points and connections drawn on the image.
    """

    TOP_FINGERS: list[int] = [8, 12, 16, 20]
    MIDDLE_FINGERS: list[int] = [6, 10, 14, 18]

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
        Initializes the hand detection system using MediaPipe Hands.

        This constructor sets up the MediaPipe Hands model with configurable parameters
        for detection and tracking, as well as drawing specifications for landmarks and connections.

        Parameters:
            mode (bool, optional): Whether to use static image mode (False for real-time tracking). Default is False.
            max_hands (int, optional): Maximum number of hands to detect. Default is 2.
            confidence_detect (float, optional): Minimum confidence threshold for hand detection. Default is 0.7.
            confidence_trace (float, optional): Minimum confidence threshold for hand tracking. Default is 0.7.
            color_points (tuple[int, int, int], optional): BGR color for drawing hand landmarks. Default is (0, 0, 255).
            color_connections (tuple[int, int, int], optional): BGR color for drawing connections between landmarks. Default is (255, 255, 255).

        Attributes:
            _mp_hands (ModuleType): MediaPipe Hands module.
            _hands (mp.solutions.hands.Hands): Configured MediaPipe Hands instance.
            _result (Any): Stores the most recent detection results (landmarks and handedness).
            _drawing (ModuleType): MediaPipe drawing utilities.
            _drawing_points (DrawingSpec): Drawing specifications for landmark points.
            _drawing_connections (DrawingSpec): Drawing specifications for connections between landmarks.
            _coordinates (list[list[Any]] | None): Stores detected coordinates of hand landmarks for each frame.
        """

        self.mode: bool = mode
        self.max_hands: int = max_hands
        self.confidence_detect: float = confidence_detect
        self.confidence_trace: float = confidence_trace
        self.color_points: tuple[int, int, int] = color_points
        self.color_connections: tuple[int, int, int] = color_connections
        self._mp_hands: ModuleType = mp.solutions.hands
        self._hands: mp.solutions.hands.Hands = self._mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=1,
            min_detection_confidence=self.confidence_detect,
            min_tracking_confidence=self.confidence_trace,
        )
        self._result: Any = None
        self._drawing: ModuleType = mp.solutions.drawing_utils
        self._drawing_points: mp.solutions.drawing_utils.DrawingSpec = (
            self._drawing.DrawingSpec(color=self.color_points)
        )
        self._drawing_connections: mp.solutions.drawing_utils.DrawingSpec = (
            self._drawing.DrawingSpec(color=self.color_connections)
        )
        self._coordinates: list[list[Any]] | None = None

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

        for tf, mf in zip(Hands.TOP_FINGERS, Hands.MIDDLE_FINGERS):
            count_fg += 1 if hand_lm[tf].y < hand_lm[mf].y else 0

        if handedness_index:
            count_fg += 1 if hand_lm[4].x < hand_lm[3].x else 0
        else:
            count_fg += 1 if hand_lm[4].x > hand_lm[3].x else 0

        return count_fg

    def _count_hand(self, hand_index: int) -> int:
        r"""
        Counts the number of extended fingers for a specific hand.

        This private method checks the detected hands and returns the count of raised fingers
        for the hand matching the given index.

        Parameters:
            hand_index (int): The index of the hand to count fingers for (e.g., 0 for left, 1 for right).

        Returns:
            int: Number of fingers detected as extended for the specified hand. Returns 0 if the hand is not detected.

        Notes:
            - Uses MediaPipe's hand landmarks stored in `self._result`.
            - Relies on the parent class method `_count_fingers` for actual finger counting logic.
            - Hand indices are determined by MediaPipe handedness classification.
        """

        if self._result.multi_hand_landmarks and self._result.multi_handedness:
            for hand_landmarks, handedness in zip(
                self._result.multi_hand_landmarks, self._result.multi_handedness
            ):
                if handedness.classification[0].index == hand_index:
                    return Hands._count_fingers(hand_landmarks, hand_index)
        return 0

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
            (cv2.Mat | ndarray): The original frame with optional drawings of detected hand landmarks
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
        color_radius: tuple[int, int, int] | None = None,
        color_landmark_id: tuple[int, int, int] | None = None,
        radius: int = 7,
        landmark_id: bool = False,
    ) -> cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]]:
        r"""
        Detects hand landmarks in a given frame, stores their coordinates, and optionally draws them.

        This method processes a frame captured from the webcam, extracts hand landmark coordinates
        using MediaPipe Hands, and draws the landmarks and their IDs on the frame if requested.

        Parameters:
            frame (cv2.Mat | ndarray): The input image/frame to process.
            drawing (bool, optional): If True, draw landmarks on the frame. Default is True.
            color_radius (tuple[int, int, int] | None, optional): Color for the landmark points in BGR.
                Defaults to black if None.
            color_landmark_id (tuple[int, int, int] | None, optional): Color for landmark IDs in BGR.
                Defaults to blue if None.
            radius (int, optional): Radius of the landmark circles. Default is 7.
            landmark_id (bool, optional): If True, draw the IDs of each landmark. Default is False.

        Returns:
            (cv2.Mat or ndarray): The processed frame with landmarks and IDs drawn if enabled.

        Notes:
            - Stores coordinates internally in `self._coordinates` as [id, x, y, hand_index].
            - Uses handedness information to slightly adjust the position of landmark IDs.
            - Handles multiple hands and landmarks per hand.
            - If color parameters are None, defaults are applied using `rgb_to_bgr`.
        """

        if color_radius is None:
            color_radius = rgb_to_bgr(0, 0, 0)

        if color_landmark_id is None:
            color_landmark_id = rgb_to_bgr(0, 0, 255)

        self._coordinates = []

        if self._result.multi_hand_landmarks and self._result.multi_handedness:
            for hand_landmarks, handedness in zip(
                self._result.multi_hand_landmarks, self._result.multi_handedness
            ):
                for id, (points, hand) in enumerate(
                    product(hand_landmarks.landmark, handedness.classification)
                ):
                    h, w, _ = frame.shape
                    cx, cy = int(points.x * w), int(points.y * h)
                    self._coordinates.append([id, cx, cy, hand.index])
                    if drawing:
                        cv2.circle(frame, (cx, cy), radius, color_radius, cv2.FILLED)
                        if landmark_id:
                            if handedness.classification[0].index == 1:
                                cv2.putText(
                                    frame,
                                    str(id),
                                    (cx - 20, cy + 20),
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
        return frame

    @property
    def count_left_point(self) -> int:
        r"""
        Returns the number of extended fingers detected on the left hand.

        Property:
            count_left_point (int): Counts the raised fingers for the left hand using the private `_count_hand` method.

        Notes:
            - Returns 0 if the left hand is not detected.
        """

        return self._count_hand(0)

    @property
    def count_right_point(self) -> int:
        r"""
        Returns the number of extended fingers detected on the right hand.

        Property:
            count_right_point (int): Counts the raised fingers for the right hand using the private `_count_hand` method.

        Notes:
            - Returns 0 if the right hand is not detected.
        """

        return self._count_hand(1)


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
        self.color_radius: tuple[int, int, int] = rgb_to_bgr(255, 255, 255)
        self.color_landmark_id: tuple[int, int, int] = rgb_to_bgr(0, 0, 0)
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
        Runs the main loop for real-time hand tracking and visualization.

        This method continuously captures frames from the webcam, processes them to detect
        hands and landmarks, optionally draws landmarks and IDs, and displays the results
        in a window. It also prints debug information if enabled.

        Steps performed:
        1. Configure the display window and webcam settings.
        2. Enter a continuous loop:
            - Capture a frame from the webcam.
            - Flip the frame horizontally for a mirror effect.
            - Detect hands and draw landmarks if enabled.
            - Detect landmark points and draw them with IDs if enabled.
            - Print coordinates for debugging if `points_debug` is True.
            - Overlay additional text via `_put_text`.
            - Display the frame in the configured window.
        3. Exit the loop if the user presses the ESC key or closes the window.
        4. Release the webcam and destroy all OpenCV windows on exit.

        Raises:
            RuntimeError: If a frame cannot be captured from the webcam.
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

            self._frame = self.search_points(
                frame=self._frame,
                drawing=self.drawing_points,
                color_radius=self.color_radius,
                color_landmark_id=self.color_landmark_id,
                radius=self.size_radius,
                landmark_id=self.landmark_id,
            )

            if self.points_debug:
                HDPrint(self._coordinates).print()

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

# python -W ignore -m src.backend.modules.math.webcam_fingers
