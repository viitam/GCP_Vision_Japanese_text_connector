import numpy as np


class FluentJapaneseVision:
    """Class for processing and analyzing Japanese text in a image."""

    def __init__(self):
        pass

    def _serialize_bounding_poly(self, bounding_poly: dict):
        """
        Serialize the bounding polygon vertices.
        要はverticesの座標がないところを0埋めしているだけだが、これ要るのか？
        boundingPoly内のverticesの要素の辞書に、xキー・yキーがないケースがあるのかと言い換えても良い。
        Args:
            bounding_poly (dict): Dict which contains list of dictionaries representing vertices.

        Returns:
            dict: Serialized bounding polygon vertices.
        """

        vertices: list[dict] = bounding_poly["vertices"]

        return {
            "vertices": [
                {"x": vertex.get("x", 0), "y": vertex.get("y", 0)}
                for vertex in vertices
            ]
        }

    def _calculate_font_size(self, bounding_poly: dict) -> int:
        """
        Calculate the font size based on bounding polygon vertices.
        Args:
            bounding_poly (dict): Google Vision APIの戻り値のboundingPoly

        Returns:
            int: Calculated font size.
        """
        vertices: list[dict] = bounding_poly.get("vertices", [])

        # 各頂点の'y'座標をリストとして取得
        y_values = [vertex["y"] for vertex in vertices if "y" in vertex]
        # y座標の最大値と最小値の差を計算
        return max(y_values) - min(y_values) if y_values else 0

    def _sort_text_sections(self, gcp_response: dict) -> list[dict]:
        """
        google vision apiの戻り値から、単語を見つけてy座標->x座標の優先順で並び替える。
        Args:
            gcp_response(dict): google vision apiからの戻り値
        Returns:
            list[dict]: 文字とその座標とフォントサイズをキーとする辞書を含むリスト
        """
        text_sections: list[dict] = gcp_response["responses"][0]["textAnnotations"]
        text_sections_sorted: list[dict] = sorted(
            [
                {
                    "description": text_section.get("description"),
                    "bounding_poly": self._serialize_bounding_poly(
                        text_section.get("boundingPoly")
                    ),
                    "font_size": self._calculate_font_size(
                        text_section.get("boundingPoly")
                    ),
                }
                for text_section in text_sections
            ],
            key=lambda w: (
                w["bounding_poly"]["vertices"][0]["y"],
                w["bounding_poly"]["vertices"][0]["x"],
            ),
        )
        return text_sections_sorted

    def _combine_sections(self, gcp_response: dict) -> list[dict]:
        """
        gcpの戻り値のbounding_polyを、近隣の文同士でくっつけた辞書をリスト化したものを出力する。
        Args:
            gcp_response(dict): gcpの戻り値のjsonを辞書にしたもの。
        Returns:
            list[dict]: _combine_adjacent_sectionsの戻り値の辞書を格納したリスト。
        """

        sections_sorted: list[dict] = self._sort_text_sections(gcp_response)[1:]
        for i in range(4):
            # sections_sortedの全ての要素を、他の要素と横に結合できないか検討する。
            # 結合によって、今までは結合できなかったものが結合できるようになっている可能性があるので、
            # 同じ手順を4回繰り返している。
            horizontally_combined_sections = []
            while sections_sorted:
                section1: dict = sections_sorted.pop(0)
                combined_section: dict = self._combine_adjacent_sections(
                    section1, sections_sorted, "horizontal"
                )
                if len(combined_section["description"]) > 1:
                    horizontally_combined_sections.append(combined_section)
            # sections_sortedの前後を並べ替える。これにより、y順でソートされたsectionsを
            # 昇降どちらの順から見るかによって、隣接するsectionとマージ可能かどうかの結果が
            # 変わることがある。一旦combineしてプールしたsectionを逆順に並べる。
            sections_sorted: list[dict] = horizontally_combined_sections[::-1]
            # ここで一旦horizontally_combined_sectionsをリセットする。
            horizontally_combined_sections = []

        # 縦方向には結合できるかの検討を1回だけ行う。
        vertically_combined_sections = []
        while sections_sorted:
            section1: dict = sections_sorted.pop(0)
            combined_section: dict = self._combine_adjacent_sections(
                section1, sections_sorted, "vertical"
            )
            if len(combined_section["description"]) > 1:
                vertically_combined_sections.append(combined_section)
        return vertically_combined_sections

    def _combine_adjacent_sections(
        self, basis_text_section: dict, sections_to_combine: list[dict], mode: str
    ) -> dict:
        """
        basis_text_sectionを、sections_to_combineに含まれる近いsectionとくっつける。
        くっつけたものを新しいsectionとする。どれともくっつかなければbasis_text_sectionがそのまま出る。
        Args:
            basis_text_section(dict): 他の単語とくっつけたいsection。
            sections_to_combine(list[dict]): くっつけられる候補のsectionのリスト
            mode(str): "horizontal"/"vertical". 選択された方法でのみ結合する。
        Returns:
            dict: basis_text_sectionに他の単語をくっつけて作った新しいsectionを表す辞書

        TODO: text_sectionをスキーマ化して可読性を高めておく。
        """
        new_description: str = basis_text_section["description"]
        new_bounding_poly: dict = basis_text_section["bounding_poly"]
        font_size: int = basis_text_section["font_size"]

        i = 0
        while i < len(sections_to_combine):
            target_text_section: dict = sections_to_combine[i]
            condition: bool = any(
                [
                    (
                        mode == "horizontal"
                        and self._can_merge_type(
                            basis_text_section, target_text_section
                        )
                        == "combine_horizontally"
                    ),
                    (
                        mode == "vertical"
                        and self._can_merge_type(
                            basis_text_section, target_text_section
                        )
                        == "combine_vertically"
                    ),
                ]
            )

            if condition:
                new_description += target_text_section["description"]
                new_bounding_poly: dict = self._create_new_bounding_poly(
                    basis_text_section, target_text_section
                )
                sections_to_combine.pop(i)
                basis_text_section: dict = {
                    "description": new_description,
                    "bounding_poly": new_bounding_poly,
                    "font_size": int(
                        np.mean([font_size, target_text_section["font_size"]])
                    ),
                }
            else:
                i += 1

        return basis_text_section

    def _can_merge_type(self, text_section1: dict, text_section2) -> str:
        """
        2つのtext_sectionを結合する際のタイプを返す。
        Args:
            text_section1(dict): 結合判定をするsection。結合時はこれが前にくる。
            text_section2(dict): 結合判定をするsection。結合時はこれが後にくる。
        Returns:
            str: "combine_horizontally"/"combine_vertically"

        """
        font_size1, font_size2 = text_section1["font_size"], text_section2["font_size"]
        vertices1, vertices2 = (
            text_section1["bounding_poly"]["vertices"],
            text_section2["bounding_poly"]["vertices"],
        )
        x1, y1, h1 = self._calculate_centroid(text_section1["bounding_poly"])
        x2, y2, h2 = self._calculate_centroid(text_section2["bounding_poly"])

        # 同じ行で、section1がsection2のすぐ左にあり、フォントサイズ比が1.5倍以内ならTrue.
        condition_horizontal: bool = all(
            [
                abs(vertices1[2]["y"] - vertices2[3]["y"]) <= (font_size1 * 0.2),
                abs(vertices1[2]["x"] - vertices2[3]["x"])
                <= (min(font_size1, font_size2) * 0.9),
                max(font_size1, font_size2) / min(font_size1, font_size2) <= 1.5,
            ]
        )

        # section1, section2の重心同士が近ければTrue. 中央揃えの改行を想定。
        condition_vertical_center: bool = all(
            [
                abs(x1 - x2) <= (max(font_size1, font_size2) * 2),
                abs(y1 - y2) <= (np.mean([h1, h2]) * 1.3),
            ]
        )

        # section1の左下と, section2の左上が近ければTrue. 左揃えの改行を想定。
        condition_vertical_left: bool = all(
            [
                abs(vertices1[0]["x"] - vertices2[0]["x"])
                <= max(font_size1, font_size2) * 2.5,
                abs(vertices1[3]["y"] - vertices2[0]["y"])
                <= max(font_size1, font_size2),
            ]
        )

        # section1の右下と, section2の右上が近ければTrue. 右揃えの改行を想定。
        # 経験則より、左揃えよりも条件が厳しくなっている。
        condition_vertical_right: bool = all(
            [
                abs(vertices1[2]["x"] - vertices2[2]["x"])
                <= max(font_size1, font_size2) * 0.5,
                abs(vertices1[3]["y"] - vertices2[0]["y"])
                <= min(font_size1, font_size2),
            ]
        )

        if condition_horizontal:
            return "combine_horizontally"

        elif any(
            [
                condition_vertical_center,
                condition_vertical_left,
                condition_vertical_right,
            ]
        ):
            return "combine_vertically"

        else:
            return None

    def _calculate_centroid(self, bounding_poly: dict) -> tuple[float, float, int]:
        """
        Calculate the centroid coordinates and height of a bounding polygon.
        Args:
            bounding_poly: Bounding polygon information.

        Returns:
            Tuple[float, float, int]: Centroid x-coordinate, y-coordinate, and height.
        """
        x_values = [vertex["x"] for vertex in bounding_poly["vertices"]]
        y_values = [vertex["y"] for vertex in bounding_poly["vertices"]]
        centroid_x = np.mean(x_values)
        centroid_y = np.mean(y_values)
        height = max(y_values) - min(y_values)
        return centroid_x, centroid_y, height

    def _create_new_bounding_poly(
        self, text_section1: dict, text_section2: dict
    ) -> dict:
        """
        Create a new bounding poly from two text_sections.
        Args:
            text_section1(dict): 結合したいtext_section
            text_section2(dict): 結合したいtext_section
        Returns:
            dict: Google Vision APIの戻り値の辞書の、boundingPolyに入るべき辞書。
        """
        vertices1 = text_section1["bounding_poly"]["vertices"]
        vertices2 = text_section2["bounding_poly"]["vertices"]

        x_values = [v["x"] for v in vertices1 + vertices2]
        y_values = [v["y"] for v in vertices1 + vertices2]

        new_bounding_poly = {
            "vertices": [
                {"x": min(x_values), "y": min(y_values)},  # 左上
                {"x": max(x_values), "y": min(y_values)},  # 右上
                {"x": max(x_values), "y": max(y_values)},  # 右下
                {"x": min(x_values), "y": max(y_values)},  # 左下
            ]
        }

        return new_bounding_poly

    def run(self, gcp_response: dict) -> list[dict]:
        """
        Run the text processing pipeline.
        Args:
            gcp_response(dict): Google Vision APIの戻り値を辞書にしたもの
        Returns:
            list[dict] : 近隣で統合を行った後のtext_section辞書が格納されたリスト
        """
        return self._combine_sections(gcp_response)  # combine