from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class ImageProcessor:
    """
    Classe responsavel pelo pre-processamento de imagens contendo equacoes matematicas.
    """

    def __init__(self, image_path: Optional[str | Path] = None):
        self.image_path = str(image_path) if image_path is not None else None
        self.original_image: np.ndarray | None = None
        self.processed_image: np.ndarray | None = None

    def load_image(self, image_path: Optional[str | Path] = None) -> np.ndarray:
        """Carrega a imagem em escala de cinza."""
        if image_path is not None:
            self.image_path = str(image_path)

        if not self.image_path:
            raise ValueError("Nenhum caminho de imagem foi informado ao ImageProcessor.")

        path = Path(self.image_path)
        if not path.exists():
            raise FileNotFoundError(f"O arquivo nao existe: {path.absolute()}")

        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Falha ao carregar a imagem (formato invalido?): {path.absolute()}")

        self.original_image = image
        return self.original_image

    def get_processed_pipeline(
        self,
        image_path: Optional[str | Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executa o pipeline basico de pre-processamento.
        Retorna: (imagem_original, imagem_binaria)
        """
        image = self.load_image(image_path)

        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        kernel = np.ones((3, 3), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.processed_image = cv2.dilate(closed, kernel, iterations=1)

        if self.original_image is None or self.processed_image is None:
            raise RuntimeError("Falha ao processar as imagens no pipeline.")

        return self.original_image, self.processed_image

    @staticmethod
    def _should_merge_boxes(
        first_box: Tuple[int, int, int, int],
        second_box: Tuple[int, int, int, int],
    ) -> bool:
        ax, ay, aw, ah = first_box
        bx, by, bw, bh = second_box

        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        horizontal_overlap = max(0, min(ax2, bx2) - max(ax, bx))
        overlap_ratio = horizontal_overlap / max(1, min(aw, bw))
        vertical_gap = max(0, max(ay, by) - min(ay2, by2))
        width_similarity = min(aw, bw) / max(1, max(aw, bw))
        center_x_distance = abs((ax + aw / 2) - (bx + bw / 2))

        first_is_stroke = ImageProcessor._looks_like_horizontal_stroke(first_box)
        second_is_stroke = ImageProcessor._looks_like_horizontal_stroke(second_box)

        return (
            first_is_stroke
            and second_is_stroke
            and overlap_ratio >= 0.8
            and vertical_gap <= max(8, int(round(max(ah, bh) * 1.6)))
            and width_similarity >= 0.6
            and center_x_distance <= max(6, int(round(max(aw, bw) * 0.35)))
        )

    @staticmethod
    def _looks_like_horizontal_stroke(box: Tuple[int, int, int, int]) -> bool:
        _, _, width, height = box
        return width >= 8 and (width / max(1, height)) >= 1.4

    @staticmethod
    def _is_likely_symbol_box(
        box: Tuple[int, int, int, int],
        contour_area: float,
    ) -> bool:
        _, _, width, height = box
        box_area = width * height
        longest_edge = max(width, height)

        return (
            (contour_area >= 24 and (width >= 6 or height >= 6))
            or longest_edge >= 12
            or box_area >= 100
        )

    def _merge_fragmented_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if len(boxes) < 2:
            return boxes

        merged_boxes = sorted(boxes, key=lambda item: (item[0], item[1]))
        did_merge = True

        while did_merge:
            did_merge = False
            next_boxes: List[Tuple[int, int, int, int]] = []
            used_indexes: set[int] = set()

            for index, current_box in enumerate(merged_boxes):
                if index in used_indexes:
                    continue

                merged_box = current_box
                for other_index in range(index + 1, len(merged_boxes)):
                    if other_index in used_indexes:
                        continue

                    candidate_box = merged_boxes[other_index]
                    if not self._should_merge_boxes(merged_box, candidate_box):
                        continue

                    mx, my, mw, mh = merged_box
                    cx, cy, cw, ch = candidate_box

                    left = min(mx, cx)
                    top = min(my, cy)
                    right = max(mx + mw, cx + cw)
                    bottom = max(my + mh, cy + ch)
                    merged_box = (left, top, right - left, bottom - top)
                    used_indexes.add(other_index)
                    did_merge = True

                used_indexes.add(index)
                next_boxes.append(merged_box)

            merged_boxes = sorted(next_boxes, key=lambda item: (item[1], item[0]))

        return merged_boxes

    @staticmethod
    def sort_boxes_reading_order(
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []

        average_height = sum(height for _, _, _, height in boxes) / len(boxes)
        line_threshold = max(6, average_height * 0.6)

        sorted_by_position = sorted(boxes, key=lambda item: (item[1], item[0]))
        lines: list[list[Tuple[int, int, int, int]]] = []

        for box in sorted_by_position:
            center_y = box[1] + box[3] / 2
            for line in lines:
                reference_center_y = line[0][1] + line[0][3] / 2
                if abs(center_y - reference_center_y) <= line_threshold:
                    line.append(box)
                    break
            else:
                lines.append([box])

        ordered_boxes: list[Tuple[int, int, int, int]] = []
        for line in lines:
            ordered_boxes.extend(sorted(line, key=lambda item: item[0]))
        return ordered_boxes

    def extract_bounding_boxes(self, binary_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detecta contornos na imagem binaria e retorna as coordenadas das bounding boxes.
        """
        if cv2.countNonZero(binary_img) == 0:
            print("AVISO: A imagem binaria esta totalmente preta!")
            return []

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            box = (x, y, w, h)
            if self._is_likely_symbol_box(box, contour_area=area):
                boxes.append(box)

        merged_boxes = self._merge_fragmented_boxes(boxes)
        return self.sort_boxes_reading_order(merged_boxes)

    def prepare_for_nn(self, roi: np.ndarray, target_size: int = 28, margin: int = 4) -> np.ndarray:
        """
        Prepara uma regiao de interesse para entrada em uma rede neural.
        """
        h, w = roi.shape
        if h == 0 or w == 0:
            return np.zeros((target_size, target_size), dtype=np.uint8)

        inner_size = max(8, target_size - (margin * 2))
        scale = inner_size / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        padded_img = np.zeros((target_size, target_size), dtype=np.uint8)
        start_y = (target_size - new_h) // 2
        start_x = (target_size - new_w) // 2
        padded_img[start_y:start_y + new_h, start_x:start_x + new_w] = resized_roi

        return padded_img


if __name__ == "__main__":
    print("ImageProcessor pronto para uso!")
