from collections import Counter
import json
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class FluentJapaneseVision:
    def __init__(self, response, save_json="data/output.json"):
        self.response = response
        self.save_json = save_json

    def serialize_bounding_poly(self, bounding_poly):
        return {"vertices": [{"x": v.x, "y": v.y} for v in bounding_poly.vertices]}

    def calculate_font_size(self, bounding_poly):
        y_values = [v.y for v in bounding_poly.vertices]
        return max(y_values) - min(y_values)

    def to_dictionary(self):
        words = sorted(
            [
                {
                    "description": word.description,
                    "bounding_poly": self.serialize_bounding_poly(word.bounding_poly),
                    "font_size": self.calculate_font_size(word.bounding_poly),
                }
                for word in self.response.text_annotations
            ],
            key=lambda w: (
                w["bounding_poly"]["vertices"][0]["y"],
                w["bounding_poly"]["vertices"][0]["x"],
            ),
        )
        result = {"words": words}
        self._save_json(result, self.save_json)
        return result

    def words_clustering(self, combined_response):
        centers = np.array(
            [
                [
                    np.mean([v["x"] for v in w["bounding_poly"]["vertices"]]),
                    np.mean([v["y"] for v in w["bounding_poly"]["vertices"]]),
                ]
                for w in combined_response
            ]
        )
        labels = DBSCAN(eps=15, min_samples=1).fit_predict(centers)

        blocks = {
            label: self._create_block(label, labels, combined_response)
            for label in set(labels)
        }

        y_min, y_max = centers[:, 1].min(), centers[:, 1].max()
        for y_start in range(int(y_min), int(y_max), 1000):
            self._plot_clusters(y_start, y_start + 1000, centers, labels, blocks)

    def combine_phrases(self):
        words = self.to_dictionary()["words"][1:]
        combined_phrases = []

        while words:
            combined_word = words.pop(0)
            font_size = combined_word["font_size"]
            i = 0

            while i < len(words):
                next_word = words[i]
                if self._can_combine_words(combined_word, next_word, font_size):
                    combined_word = self._merge_words(combined_word, next_word)
                    words.pop(i)
                else:
                    i += 1

            if len(combined_word["description"]) > 1:
                combined_phrases.append(combined_word)

        self._save_json(combined_phrases, "data/word_result.json")
        return combined_phrases

    def combine_lines(self):
        sentences = self.combine_phrases()
        combined_lines = []

        while sentences:
            line = sentences.pop(0)
            font_size = line["font_size"]
            i = 0

            while i < len(sentences):
                next_line = sentences[i]
                if self._can_combine_lines(line, next_line, font_size):
                    line = self._merge_words(line, next_line)
                    sentences.pop(i)
                else:
                    i += 1

            combined_lines.append(line)

        self._save_json(combined_lines, "data/line_result.json")
        return combined_lines

    def process_text(self):
        self.combine_phrases()
        self.combine_lines()

    def _save_json(self, data, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def _create_block(self, label, labels, response):
        block_points = [
            response[i]["bounding_poly"]["vertices"]
            for i, lbl in enumerate(labels)
            if lbl == label
        ]
        all_points = np.concatenate(
            [[list(v.values()) for v in p] for p in block_points]
        )
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        return {
            "min_x": min_x,
            "min_y": min_y,
            "max_x": max_x,
            "max_y": max_y,
            "words": [
                response[i]["description"]
                for i, lbl in enumerate(labels)
                if lbl == label
            ],
        }

    def _plot_clusters(self, y_start, y_end, centers, labels, blocks):
        plt.figure(figsize=(8, 6))
        colors = ["r", "g", "b", "c", "m", "y", "k"]
        plt.ylim(y_start, y_end)
        plt.gca().invert_yaxis()

        for i, center in enumerate(centers):
            if y_start <= center[1] < y_end:
                plt.scatter(center[0], center[1], color=colors[labels[i] % len(colors)])

        for block in blocks.values():
            if y_start <= block["min_y"] < y_end or y_start <= block["max_y"] < y_end:
                plt.gca().add_patch(
                    plt.Rectangle(
                        (block["min_x"], block["min_y"]),
                        block["max_x"] - block["min_x"],
                        block["max_y"] - block["min_y"],
                        fill=False,
                        edgecolor=colors[labels[i] % len(colors)],
                        linewidth=2,
                    )
                )

        plt.savefig(
            f"data/words_clustering_y_{y_start}_{y_end}.png", bbox_inches="tight"
        )
        plt.close()

    def _can_combine_words(self, word1, word2, font_size):
        vertices1, vertices2 = (
            word1["bounding_poly"]["vertices"],
            word2["bounding_poly"]["vertices"],
        )
        return (
            abs(vertices1[2]["y"] - vertices2[3]["y"]) <= font_size / 5
            and abs(vertices1[2]["x"] - vertices2[3]["x"]) <= font_size * 0.9
        )

    def _can_combine_lines(self, line1, line2, font_size):
        x1, y1 = self._calculate_centroid(line1["bounding_poly"])
        x2, y2 = self._calculate_centroid(line2["bounding_poly"])
        return abs(x1 - x2) <= font_size * 2 and abs(y1 - y2) <= font_size * 1.5

    def _merge_words(self, word1, word2):
        x_vals = [
            v["x"]
            for v in word1["bounding_poly"]["vertices"]
            + word2["bounding_poly"]["vertices"]
        ]
        y_vals = [
            v["y"]
            for v in word1["bounding_poly"]["vertices"]
            + word2["bounding_poly"]["vertices"]
        ]
        return {
            "description": word1["description"] + word2["description"],
            "bounding_poly": {
                "vertices": [
                    {"x": min(x_vals), "y": min(y_vals)},
                    {"x": max(x_vals), "y": min(y_vals)},
                    {"x": max(x_vals), "y": max(y_vals)},
                    {"x": min(x_vals), "y": max(y_vals)},
                ]
            },
            "font_size": (word1["font_size"] + word2["font_size"]) / 2,
        }

    def _calculate_centroid(self, bounding_poly):
        x_vals = [v["x"] for v in bounding_poly["vertices"]]
        y_vals = [v["y"] for v in bounding_poly["vertices"]]
        return sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals)

    def run(self):
        combined_lines = self.combine_lines()
        self.words_clustering(combined_lines)
