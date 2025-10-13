import cv2
import mediapipe as mp
from itertools import product
from typing import Any
from dataclasses import dataclass
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


@dataclass(slots=True)
class WindowConfig:
    r"""
    Configuration for the main application window and camera capture parameters.

    Attributes:
        name (str): Window title to display during visualization.
        width (int): Display window width in pixels.
        height (int): Display window height in pixels.
        width_frame (int): Width of the captured frame from the camera.
        height_frame (int): Height of the captured frame from the camera.
        fps (int): Desired frames per second for video capture.
        gain (int): Camera gain level (brightness amplification factor).
    """

    name: str = "Count Fingers"
    width: int = 1280
    height: int = 720
    width_frame: int = 1920
    height_frame: int = 1080
    fps: int = 30
    gain: int = 0


@dataclass(slots=True)
class ColorScheme:
    r"""
    Color configuration for drawing elements in the hand-tracking interface.

    Attributes:
        points (tuple[int, int, int]): Color for individual landmark points.
        connections (tuple[int, int, int]): Color for lines connecting landmarks.
        radius (tuple[int, int, int]): Color for the radius circle drawn around landmarks.
        landmark_id (tuple[int, int, int]): Color for text showing landmark IDs.
        rectangle (tuple[int, int, int]): Color for the bounding rectangle around the hand.
        left (tuple[int, int, int]): Color for information or markers related to the left hand.
        right (tuple[int, int, int]): Color for information or markers related to the right hand.
        total (tuple[int, int, int]): Color for the total finger count text or indicators.
    """

    points: tuple[int, int, int] = rgb_to_bgr(255, 0, 0)
    connections: tuple[int, int, int] = rgb_to_bgr(0, 0, 0)
    radius: tuple[int, int, int] = rgb_to_bgr(255, 255, 255)
    landmark_id: tuple[int, int, int] = rgb_to_bgr(0, 0, 0)
    rectangle: tuple[int, int, int] = rgb_to_bgr(0, 0, 0)
    left: tuple[int, int, int] = rgb_to_bgr(255, 255, 255)
    right: tuple[int, int, int] = rgb_to_bgr(255, 255, 255)
    total: tuple[int, int, int] = rgb_to_bgr(255, 255, 255)

    def __post_init__(self):
        """
        Converts all color attributes to BGR format after initialization.

        This ensures consistency in color representation for OpenCV,
        which uses BGR instead of RGB color order.
        """

        self.points = rgb_to_bgr(*self.points)
        self.connections = rgb_to_bgr(*self.connections)
        self.radius = rgb_to_bgr(*self.radius)
        self.landmark_id = rgb_to_bgr(*self.landmark_id)
        self.rectangle = rgb_to_bgr(*self.rectangle)
        self.left = rgb_to_bgr(*self.left)
        self.right = rgb_to_bgr(*self.right)
        self.total = rgb_to_bgr(*self.total)


@dataclass(slots=True)
class HandConfig:
    r"""
    Configuration for the hand detection and drawing parameters.

    Attributes:
        mode (bool): Whether to use static image mode (True) or video stream mode (False).
        max_hands (int): Maximum number of hands to detect simultaneously.
        confidence_detect (float): Minimum confidence threshold for hand detection.
        confidence_trace (float): Minimum confidence threshold for hand tracking.
        drawing_hands (bool): Whether to draw detected hand outlines and landmarks.
        drawing_points (bool): Whether to draw individual landmark points.
        landmark_id (bool): Whether to display landmark ID numbers on the image.
        size_radius (int): Radius size for drawing landmark circles.
    """

    mode: bool = False
    max_hands: int = 2
    confidence_detect: float = 0.7
    confidence_trace: float = 0.7
    drawing_hands: bool = True
    drawing_points: bool = True
    landmark_id: bool = True
    size_radius: int = 7


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

    @property
    def me_hands(self) -> ModuleType:
        r"""
        Get the MediaPipe hands module.

        Returns:
            ModuleType: The MediaPipe `hands` module used for hand detection.
        """

        return self._mp_hands

    @property
    def hands(self) -> mp.solutions.hands.Hands:
        r"""
        Get the MediaPipe Hands object.

        Returns:
            mp.solutions.hands.Hands: The initialized MediaPipe Hands instance
            used to process video frames.
        """

        return self._hands

    @property
    def result(self) -> Any:
        r"""
        Get the latest hand detection result.

        Returns:
            Any: The output from MediaPipe Hands after processing the current frame.
        """

        return self._result

    @property
    def drawing(self) -> ModuleType:
        r"""
        Get the MediaPipe drawing utilities module.

        Returns:
            ModuleType: The `drawing_utils` module used for rendering landmarks
            and connections.
        """

        return self._drawing

    @property
    def drawing_points(self) -> mp.solutions.drawing_utils.DrawingSpec:
        r"""
        Get the DrawingSpec for hand landmark points.

        Returns:
            mp.solutions.drawing_utils.DrawingSpec: The drawing configuration
            used for rendering individual hand landmarks.
        """

        return self._drawing_points

    @property
    def drawing_connections(self) -> mp.solutions.drawing_utils.DrawingSpec:
        r"""
        Get the DrawingSpec for hand connections.

        Returns:
            mp.solutions.drawing_utils.DrawingSpec: The drawing configuration
            used for rendering connections between landmarks.
        """

        return self._drawing_connections

    @property
    def coordinates(self) -> list[list[Any]] | None:
        r"""
        Get the list of detected hand coordinates.

        Returns:
            (list[list[Any]] | None): A nested list of coordinates for each detected hand,
            or None if no hands are detected.
        """

        return self._coordinates

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
                    self._coordinates.append([id, cx, cy, hand.label.upper()])
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
    WebCamActivate class for managing real-time hand detection via webcam.

    This class extends the Hands class and provides a complete system for
    capturing video from a webcam, detecting hands and landmarks in real-time,
    overlaying visual information, and optionally providing debug output.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize the webcam capture system and configure hand detection settings.

        This constructor sets up the webcam, window parameters, color schemes,
        and hand detection configuration. It also initializes internal attributes
        used for frame processing and debugging.

        Components initialized:
            - Webcam:
                Opens the default webcam using OpenCV with the DirectShow backend,
                setting up attributes for frame capture (`_cap`), capture status (`_ret`),
                and the current frame (`_frame`). Also defines `_webcamrun` to control
                the main loop execution. Raises an `IOError` if the webcam cannot be accessed.

            - Window and Visual Settings:
                `window` defines frame dimensions, FPS, and gain.
                `colors` defines the color scheme for points, hands, and overlays.
                `hands_cfg` defines hand detection parameters such as mode, max hands,
                detection confidence, and tracking confidence.

            - Debug:
                `points_debug` enables printing of landmark coordinates for debugging.

            - Superclass Initialization:
                Passes hand detection parameters and color configuration to the parent
                class for proper setup.

        Raises:
            IOError: If the webcam cannot be opened.
        """

        self._cap: cv2.VideoCapture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self._ret: bool = False
        self._frame: cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]] = None
        self._webcamrun: bool = True
        self.points_debug: bool = True
        self.window: WindowConfig = WindowConfig()
        self.colors: ColorScheme = ColorScheme()
        self.hands_cfg: HandConfig = HandConfig()

        super().__init__(
            mode=self.hands_cfg.mode,
            max_hands=self.hands_cfg.max_hands,
            confidence_detect=self.hands_cfg.confidence_detect,
            confidence_trace=self.hands_cfg.confidence_trace,
            color_points=self.colors.points,
            color_connections=self.colors.connections,
        )

        if not self._cap.isOpened():
            raise IOError("Unable to access webcam.")

    @property
    def cap(self) -> cv2.VideoCapture:
        r"""
        Get the internal OpenCV VideoCapture object.

        This property provides read-only access to the webcam capture instance
        used for acquiring frames from the camera.

        Returns:
            cv2.VideoCapture: The VideoCapture object managing webcam input.
        """

        return self._cap

    @property
    def ret(self) -> bool:
        r"""
        Get the status of the last frame capture.

        This property provides read-only access to the result of the most recent
        frame acquisition attempt, indicating whether it was successful.

        Returns:
            bool: True if the last frame was successfully captured, False otherwise.
        """

        return self._ret

    @property
    def frame(self) -> cv2.Mat | ndarray[Any, dtype[integer[Any] | floating[Any]]]:
        r"""
        Get the current video frame.

        This property provides read-only access to the latest captured frame,
        represented either as an OpenCV Mat or a NumPy array.

        Returns:
            (cv2.Mat | ndarray): The most recent video frame.
        """

        return self._frame

    @property
    def webcamrun(self) -> bool:
        r"""
        Get the webcam loop running status.

        This property provides read-only access to the flag indicating whether
        the main webcam loop is currently active.

        Returns:
            bool: True if the main webcam loop is running, False otherwise.
        """

        return self._webcamrun

    def _config_window(self) -> None:
        r"""
        Configures the OpenCV display window.

        - Creates a resizable window with the name specified in `self.window_name`.
        - Sets the window dimensions to `self.window_width` x `self.window_height`.

        Notes
        -----
        - Should be called before displaying any frames with `cv2.imshow`.
        """

        cv2.namedWindow(self.window.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window.name, self.window.width, self.window.height)

    def _put_text(
        self,
    ) -> None:
        r"""
        Overlay hand count information on the current video frame.

        This method draws a semi-transparent rectangle in the top-left corner of
        the frame and displays the number of fingers detected for the left hand,
        right hand, and total count. Colors for each hand and the total are
        taken from the configured color scheme.

        Details:
            - Left-hand count is shown in the configured left-hand color.
            - Right-hand count is shown in the configured right-hand color.
            - Total fingers count is shown in the configured total color.
            - Uses OpenCV's FONT_HERSHEY_SIMPLEX for text rendering with fixed font size and thickness.
        """

        total_count: int = self.count_left_point + self.count_right_point

        cv2.rectangle(self._frame, (5, 5), (230, 100), self.colors.rectangle, -1)

        cv2.putText(
            self._frame,
            f"Left Hand: {self.count_left_point}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.colors.left,
            2,
        )
        cv2.putText(
            self._frame,
            f"Right Hand: {self.count_right_point}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.colors.right,
            2,
        )
        cv2.putText(
            self._frame,
            f"Total Fingers: {total_count}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            self.colors.total,
            2,
        )

    def _config_webcam(self) -> None:
        r"""
        Configure the webcam capture settings.

        Sets the frame width, frame height, frames per second (FPS), and gain
        for the webcam using values defined in the `window` configuration.
        """

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.window.width_frame)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.window.height_frame)
        self._cap.set(cv2.CAP_PROP_FPS, self.window.fps)
        self._cap.set(cv2.CAP_PROP_GAIN, self.window.gain)

    def mainloop(self) -> None:
        r"""
        Run the main loop for webcam capture and real-time hand processing.

        Initializes the webcam and window configurations, then continuously captures
        and processes frames for hand and landmark detection. The loop overlays visual
        elements, updates text information, and displays the resulting frame until the
        user presses the ESC key or closes the window.

        Steps performed:
            1. Capture a frame from the webcam.
            2. Flip the frame horizontally for a mirror effect.
            3. Detect and optionally draw hands.
            4. Detect and optionally draw landmarks and points.
            5. Optionally print detected coordinates for debugging.
            6. Overlay hand count and other information on the frame.
            7. Display the processed frame in the application window.
            8. Exit the loop on ESC press or window closure.

        Raises:
            RuntimeError: If a frame cannot be captured from the webcam.
        """

        self._config_window()

        self._config_webcam()

        while self._webcamrun:

            self._ret, self._frame = self._cap.read()
            if not self._ret:
                raise RuntimeError("Failed to capture frame.")

            self._frame = cv2.flip(self._frame, 1)

            self._frame = self.search_hands(
                frame=self._frame, drawing=self.hands_cfg.drawing_hands
            )

            self._frame = self.search_points(
                frame=self._frame,
                drawing=self.hands_cfg.drawing_points,
                color_radius=self.colors.radius,
                color_landmark_id=self.colors.landmark_id,
                radius=self.hands_cfg.size_radius,
                landmark_id=self.hands_cfg.landmark_id,
            )

            if self.points_debug:
                HDPrint(self._coordinates).print()

            self._put_text()

            cv2.imshow(self.window.name, self._frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            if cv2.getWindowProperty(self.window.name, cv2.WND_PROP_VISIBLE) < 1:
                break

        self._cap.release()
        cv2.destroyAllWindows()

    def stop(self) -> None:
        r"""
        Stop the webcam main loop.

        Sets the internal `_webcamrun` flag to False, which causes the main loop
        in `mainloop()` to exit gracefully on the next iteration.

        Parameters:
            None

        Returns:
            None
        """

        self._webcamrun = False


if __name__ == "__main__":

    webcam = WebCamActivate()
    webcam.mainloop()


# python -W ignore -m src.backend.modules.math.webcam_fingers
