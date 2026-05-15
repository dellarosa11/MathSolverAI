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
        self.processing_debug: dict[str, np.ndarray] = {}

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

    @staticmethod
    def _ensure_odd(value: int) -> int:
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _filter_horizontal_line_components(
        mask: np.ndarray,
        *,
        min_width: int,
        max_height: int,
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)

        for label_index in range(1, num_labels):
            x, y, width, height, area = stats[label_index]
            if area <= 0:
                continue
            if width < min_width or height > max_height:
                continue
            filtered[labels == label_index] = 255

        return filtered

    def _remove_notebook_lines(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        dark_pixels = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )

        horizontal_kernel_width = max(24, width // 8)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))
        horizontal_lines = cv2.morphologyEx(dark_pixels, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        filtered_lines = self._filter_horizontal_line_components(
            horizontal_lines,
            min_width=max(28, int(width * 0.30)),
            max_height=max(5, height // 24),
        )
        line_mask = cv2.dilate(filtered_lines, np.ones((3, 3), np.uint8), iterations=1)
        cleaned = cv2.inpaint(image, line_mask, 3, cv2.INPAINT_TELEA)

        return cleaned, line_mask

    def _build_processing_debug(
        self,
        image_path: Optional[str | Path] = None,
    ) -> dict[str, np.ndarray]:
        image = self.load_image(image_path)

        denoised = cv2.fastNlMeansDenoising(image, None, h=9, templateWindowSize=7, searchWindowSize=21)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        line_removed, line_mask = self._remove_notebook_lines(contrast)

        background_size = self._ensure_odd(max(31, (min(image.shape[:2]) // 12) | 1))
        background = cv2.GaussianBlur(line_removed, (background_size, background_size), 0)
        normalized = cv2.divide(line_removed, np.maximum(background, 1), scale=255)

        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            17,
            8,
        )

        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        final_binary = cv2.dilate(closed, kernel, iterations=1)

        self.processed_image = final_binary
        self.processing_debug = {
            "original": image.copy(),
            "denoised": denoised,
            "contrast": contrast,
            "line_mask": line_mask,
            "line_removed": line_removed,
            "normalized": normalized,
            "blurred": blurred,
            "threshold": thresh,
            "opened": opened,
            "closed": closed,
            "final_binary": final_binary,
        }
        return self.processing_debug

    def get_processed_pipeline(
        self,
        image_path: Optional[str | Path] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executa o pipeline basico de pre-processamento.
        Retorna: (imagem_original, imagem_binaria)
        """
        debug = self._build_processing_debug(image_path)

        if self.original_image is None or self.processed_image is None:
            raise RuntimeError("Falha ao processar as imagens no pipeline.")

        return self.original_image, self.processed_image

    def get_processing_debug(
        self,
        image_path: Optional[str | Path] = None,
    ) -> dict[str, np.ndarray]:
        """
        Retorna as etapas intermediarias do tratamento da imagem.
        """
        debug = self._build_processing_debug(image_path)
        return {name: stage.copy() for name, stage in debug.items()}

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
    def _looks_like_small_dot(box: Tuple[int, int, int, int]) -> bool:
        _, _, width, height = box
        return width <= 12 and height <= 12

    @staticmethod
    def _looks_like_division_core(box: Tuple[int, int, int, int]) -> bool:
        _, _, width, height = box
        aspect_ratio = width / max(1, height)
        return width >= 12 and aspect_ratio >= 1.4 and height <= 18

    @staticmethod
    def is_fraction_bar_box(box: Tuple[int, int, int, int]) -> bool:
        _, _, width, height = box
        return width >= 18 and height <= 12 and (width / max(1, height)) >= 2.4

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

    @staticmethod
    def _merge_boxes(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        left = min(x for x, _, _, _ in boxes)
        top = min(y for _, y, _, _ in boxes)
        right = max(x + w for x, _, w, _ in boxes)
        bottom = max(y + h for _, y, _, h in boxes)
        return left, top, right - left, bottom - top

    @staticmethod
    def _vertical_projection_split_ranges(
        roi: np.ndarray,
        *,
        min_segment_width: int,
    ) -> list[tuple[int, int]]:
        projection = (roi > 0).sum(axis=0).astype(np.int32)
        width = int(projection.shape[0])
        if width < max(10, min_segment_width * 2):
            return [(0, width)]

        margin = max(2, width // 10)
        if width <= margin * 2 + 2:
            return [(0, width)]

        interior = projection[margin: width - margin]
        if interior.size == 0:
            return [(0, width)]

        max_value = int(interior.max(initial=0))
        if max_value <= 0:
            return [(0, width)]

        threshold = max(1, int(round(max_value * 0.2)))
        low_columns = [index + margin for index, value in enumerate(interior) if int(value) <= threshold]
        if not low_columns:
            return [(0, width)]

        groups: list[list[int]] = []
        current_group: list[int] = [low_columns[0]]
        for column in low_columns[1:]:
            if column == current_group[-1] + 1:
                current_group.append(column)
            else:
                groups.append(current_group)
                current_group = [column]
        groups.append(current_group)

        split_positions: list[int] = []
        for group in groups:
            split_column = group[len(group) // 2]
            left_width = split_column
            right_width = width - split_column
            if left_width >= min_segment_width and right_width >= min_segment_width:
                split_positions.append(split_column)

        if not split_positions:
            return [(0, width)]

        ranges: list[tuple[int, int]] = []
        start = 0
        for split_column in split_positions:
            if split_column - start >= min_segment_width:
                ranges.append((start, split_column))
                start = split_column
        if width - start >= min_segment_width:
            ranges.append((start, width))

        if len(ranges) <= 1:
            return [(0, width)]
        return ranges

    def _split_wide_boxes(
        self,
        binary_img: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        split_boxes: list[Tuple[int, int, int, int]] = []

        for box in boxes:
            x, y, width, height = box
            if height < 12 or width < max(28, int(round(height * 1.35))) or self.is_fraction_bar_box(box):
                split_boxes.append(box)
                continue

            roi = binary_img[y:y + height, x:x + width]
            ranges = self._vertical_projection_split_ranges(
                roi,
                min_segment_width=max(8, height // 3),
            )
            if len(ranges) <= 1:
                split_boxes.append(box)
                continue

            for start, end in ranges:
                sub_roi = roi[:, start:end]
                if cv2.countNonZero(sub_roi) == 0:
                    continue
                contour_points = cv2.findNonZero(sub_roi)
                if contour_points is None:
                    continue
                sub_x, sub_y, sub_w, sub_h = cv2.boundingRect(contour_points)
                split_boxes.append((x + start + sub_x, y + sub_y, sub_w, sub_h))

        return split_boxes

    def _merge_division_boxes(
        self,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int, int, int]]:
        if len(boxes) < 3:
            return boxes

        ordered = sorted(boxes, key=lambda item: (item[1], item[0]))
        consumed_indexes: set[int] = set()
        merged: list[Tuple[int, int, int, int]] = []

        for index, box in enumerate(ordered):
            if index in consumed_indexes or not self._looks_like_division_core(box):
                continue

            x, y, width, height = box
            center_x = x + width / 2
            center_y = y + height / 2
            best_above: tuple[int, Tuple[int, int, int, int]] | None = None
            best_below: tuple[int, Tuple[int, int, int, int]] | None = None

            for other_index, other_box in enumerate(ordered):
                if other_index == index or other_index in consumed_indexes:
                    continue
                if not self._looks_like_small_dot(other_box):
                    continue

                ox, oy, ow, oh = other_box
                other_center_x = ox + ow / 2
                other_center_y = oy + oh / 2
                if abs(other_center_x - center_x) > max(10, width * 0.35):
                    continue

                if other_center_y < center_y:
                    if center_y - other_center_y > max(20, height * 2.4):
                        continue
                    if best_above is None or other_center_y > (best_above[1][1] + best_above[1][3] / 2):
                        best_above = (other_index, other_box)
                elif other_center_y > center_y:
                    if other_center_y - center_y > max(20, height * 2.4):
                        continue
                    if best_below is None or other_center_y < (best_below[1][1] + best_below[1][3] / 2):
                        best_below = (other_index, other_box)

            if best_above and best_below:
                merged_box = self._merge_boxes([box, best_above[1], best_below[1]])
                consumed_indexes.update({index, best_above[0], best_below[0]})
                merged.append(merged_box)
        for index, box in enumerate(ordered):
            if index not in consumed_indexes:
                merged.append(box)

        return sorted(merged, key=lambda item: (item[1], item[0]))

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

        contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            box = (x, y, w, h)
            if self._is_likely_symbol_box(box, contour_area=area):
                boxes.append(box)

        split_boxes = self._split_wide_boxes(binary_img, boxes)
        division_merged_boxes = self._merge_division_boxes(split_boxes)
        merged_boxes = self._merge_fragmented_boxes(division_merged_boxes)
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
