import contextlib
import io
import unittest

from chaldea.chain import (
    ChainNode,
    CharaClassifyChain,
    FaceClassifyChain,
    NarrowFigureClassifyChain,
    CommandsClassifyChain,
    StatusClassifyChain,
    ChaldeaJsonMerger,
    CharaFacePairer,
    AvatarDetectChain,
    ExpandClipChain,
    OverlayChain,
    FrameChain,
    LabelChain,
    StatusChain,
    StarsChain,
    ClassChain,
    PostChain,
    BackgroundClipChain,
    FaceMixChain,
    CopyChain,
    NarrowFigureFrameChain,
    NarrowFigureBorderChain,
    NarrowFigureLabelChain,
    NarrowFigureStarsChain,
    NarrowFigureClassChain,
    NarrowFigureCraftChain,
)
from chaldea.composite_support import (
    FG_DIR,
    FRAME_CONFIG,
    BORDER_CONFIG,
    LABEL_LIST,
    STARS_CONFIG,
    CLASS_CONFIG,
)
from chaldea.config import RES_DIR, TEMP_DIR


class TestUnit(unittest.TestCase):
    def test_chain_node(self):
        img_path = RES_DIR / "test" / "133019207_p0.jpg"
        node = ChainNode(img_path)
        node.save(TEMP_DIR / "output.png")

    def test_expand_clip_chain(self):
        chain = ExpandClipChain(
            target_img=RES_DIR / "test" / "哈贝特洛特(Pretender)一破.png",
            rect=(190, 139, 324, 285),
            ratio=160 / 160,
            min_scale=0.85,
            max_scale=1.5,
            steps=5,
            output_dir=TEMP_DIR / "481",
            # mask_img=RES_DIR / "mask" / "BGO头像裁剪掩码.png",
        )
        nodes = chain.output()

        print(f"\nExpandClipChain produced {len(nodes)} clips:")
        for node in nodes:
            print(f"  {node.image_path} size={node.image.size}")

    def test_frame_chain(self):
        chain = FrameChain(
            name="金",
            frame=RES_DIR / "foreground" / "L0_金框.png",
            expand_chain=[
                ChainNode(RES_DIR / "test" / "哈贝特洛特(Pretender)一破_bg_0.png"),
            ],
            output_dir=TEMP_DIR / "481",
            mask=RES_DIR / "mask" / "BGO头像裁剪掩码.png",
        )
        nodes = chain.output()

        print(f"\nFrameChain produced {len(nodes)} frames:")
        for node in nodes:
            print(f"  {node.image_path} size={node.image.size}")

        self.assertEqual(len(nodes), 1)
        for node in nodes:
            self.assertTrue(node.image_path.exists())
            self.assertEqual(node.image.mode, "RGBA")

    def test_label_chain(self):
        chain = LabelChain(
            frame_chain=[
                ChainNode(RES_DIR / "test" / "哈贝特洛特(Pretender)一破_bg_0_金_0.png"),
            ],
            label_list=[
                RES_DIR / "foreground" / "L1_金标.png",
                RES_DIR / "foreground" / "L1_满级金标.png",
            ],
            output_dir=TEMP_DIR / "481",
        )
        nodes = chain.output()

        print(f"\nLabelChain produced {len(nodes)} labels:")
        for node in nodes:
            print(f"  {node.image_path} size={node.image.size}")

        self.assertEqual(len(nodes), 2)
        for node in nodes:
            self.assertTrue(node.image_path.exists())
            self.assertEqual(node.image.mode, "RGBA")

    def test_status_chain(self):
        chain = StatusChain(
            label_chain=[
                ChainNode(RES_DIR / "test" / "哈贝特洛特(Pretender)一破_bg_0_金_0.png"),
            ],
            status_list=[
                RES_DIR / "foreground" / "L1_满破标.png",
            ],
            output_dir=TEMP_DIR / "481",
        )
        nodes = chain.output()

        print(f"\nStatusChain produced {len(nodes)} statuses:")
        for node in nodes:
            print(f"  {node.image_path} size={node.image.size}")

        self.assertEqual(len(nodes), 1)
        for node in nodes:
            self.assertTrue(node.image_path.exists())
            self.assertEqual(node.image.mode, "RGBA")

    def test_stars_chain(self):
        chain = StarsChain(
            status_chain=[
                ChainNode(RES_DIR / "test" / "哈贝特洛特(Pretender)一破_bg_0_金_0.png"),
            ],
            output_dir=TEMP_DIR / "481",
            stars_list=[
                RES_DIR / "foreground" / "L1_4星.png",
            ],
            rank_name="4星",
        )
        nodes = chain.output()

        print(f"\nStarsChain produced {len(nodes)} stars:")
        for node in nodes:
            print(f"  {node.image_path} size={node.image.size}")

        self.assertEqual(len(nodes), 1)
        for node in nodes:
            self.assertTrue(node.image_path.exists())
            self.assertEqual(node.image.mode, "RGBA")

    def test_class_chain(self):
        chain = ClassChain(
            stars_chain=[
                ChainNode(RES_DIR / "test" / "哈贝特洛特(Pretender)一破_bg_0_金_0.png"),
            ],
            class_list=[
                RES_DIR / "foreground" / "金卡Berserker.png",
            ],
            class_name="金卡Berserker",
            output_dir=TEMP_DIR / "481",
        )
        nodes = chain.output()

        print(f"\nClassChain produced {len(nodes)} classes:")
        for node in nodes:
            print(f"  {node.image_path} size={node.image.size}")

        self.assertEqual(len(nodes), 1)
        for node in nodes:
            self.assertTrue(node.image_path.exists())
            self.assertEqual(node.image.mode, "RGBA")

    def test_post_chain(self):
        input_nodes = [ChainNode(RES_DIR / "test" / "哈贝特洛特(Pretender)一破_bg_0_金_0.png")]
        chains = [
            LabelChain(
                frame_chain=input_nodes,
                label_list=[
                    RES_DIR / "foreground" / "L1_金标.png",
                    RES_DIR / "foreground" / "L1_满级金标.png",
                ],
                output_dir=TEMP_DIR / "481",
            ),
            StatusChain(
                label_chain=input_nodes,
                status_list=[RES_DIR / "foreground" / "L1_满破标.png"],
                output_dir=TEMP_DIR / "481",
            ),
            ClassChain(
                stars_chain=input_nodes,
                class_list=[RES_DIR / "foreground" / "金卡Berserker.png"],
                class_name="金卡Berserker",
                output_dir=TEMP_DIR / "481",
            ),
        ]
        nodes = PostChain(chains).output()

        print(f"\nPostChain merged {len(nodes)} nodes from {len(chains)} chains:")
        for node in nodes:
            print(f"  {node.image_path.name} size={node.image.size}")

        self.assertEqual(len(nodes), 4)
        self.assertEqual(len({node.image_path for node in nodes}), len(nodes))
        for node in nodes:
            self.assertTrue(node.image_path.exists())

    def test_overlay_chain_name_conflict(self):
        OverlayChain._produced.clear()
        input_nodes = [ChainNode(RES_DIR / "test" / "哈贝特洛特(Pretender)一破_bg_0_金_0.png")]
        chains = [
            LabelChain(
                frame_chain=input_nodes,
                label_list=[RES_DIR / "foreground" / "L1_金标.png"],
                output_dir=TEMP_DIR / "481",
            ),
            LabelChain(
                frame_chain=input_nodes,
                label_list=[RES_DIR / "foreground" / "L1_满级金标.png"],
                output_dir=TEMP_DIR / "481",
            ),
        ]

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            nodes = PostChain(chains).output()
        log = buffer.getvalue()

        print(f"\nOverlayChain name conflict log:\n{log}")

        # 两条链各自只有 1 个图层，序号都从 0 起，产物同名互相顶掉
        self.assertEqual(log.count("输出文件重名"), 1)
        self.assertEqual(len(nodes), 2)
        self.assertEqual(len({node.image_path for node in nodes}), 1)


    def test_background_clip_chain(self):
        chain = BackgroundClipChain(output_size=(280, 280))
        nodes = chain.output()

        print(f"\nBackgroundClipChain produced {len(nodes)} clips:")
        for node in nodes[:3]:
            print(f"  size={node.image.size} mode={node.image.mode} image_path={node.image_path}")
        print(f"  共 {len(nodes)} 片，全部只在内存中传递")

        self.assertGreater(len(nodes), 0)
        for node in nodes:
            self.assertIsNone(node.image_path)
            self.assertEqual(node.image.size, (280, 280))
            self.assertEqual(node.image.mode, "RGBA")

    def test_face_mix_chain(self):
        status_nodes = StatusClassifyChain(RES_DIR / "assets" / "1").output()[:2]
        bg_nodes = BackgroundClipChain(output_size=(280, 280)).output()[:3]

        chain = FaceMixChain(status_nodes=status_nodes, bg_nodes=bg_nodes,
                             output_dir=TEMP_DIR / "481")
        nodes = chain.output()

        print(f"\nFaceMixChain produced {len(nodes)} mesh images:")
        for node in nodes:
            print(f"  {node.image_path.name} size={node.image.size}")

        self.assertEqual(len(nodes), len(status_nodes) * len(bg_nodes))
        for node in nodes:
            self.assertTrue(node.image_path.exists())
            self.assertEqual(node.image.size, (280, 280))
            self.assertEqual(node.image.mode, "RGBA")

    def test_copy_chain(self):
        face_nodes = FaceClassifyChain(RES_DIR / "assets" / "1").output()
        chain = CopyChain(input_nodes=face_nodes, output_dir=TEMP_DIR / "481")
        nodes = chain.output()

        print(f"\nCopyChain copied {len(nodes)} faces:")
        for node in nodes:
            print(f"  {node.image_path} size={node.image.size}")

        self.assertEqual(len(nodes), len(face_nodes))
        for node, src in zip(nodes, face_nodes):
            self.assertTrue(node.image_path.exists())
            self.assertEqual(node.image_path.name, src.image_path.name)
            self.assertEqual(node.image_path.stat().st_size, src.image_path.stat().st_size)

    def test_narrow_figure_chains(self):
        output_dir = TEMP_DIR / "support_test"
        figures = NarrowFigureClassifyChain(RES_DIR / "assets" / "1").output()
        self.assertGreater(len(figures), 0)

        frame_chain = NarrowFigureFrameChain(
            name="金卡_金",
            frame=FRAME_CONFIG["金卡"]["金"],
            narrow_figure_chain=figures,
            output_dir=output_dir,
        ).output()
        border_chain = NarrowFigureBorderChain(
            frame_chain=frame_chain,
            border_list=BORDER_CONFIG["金卡"]["金"],
            output_dir=output_dir,
        ).output()
        label_chain = NarrowFigureLabelChain(
            border_chain=border_chain,
            label_list=LABEL_LIST,
            output_dir=output_dir,
        ).output()
        stars_chain = NarrowFigureStarsChain(
            label_chain=label_chain,
            stars_list=STARS_CONFIG["金卡"]["金"],
            rank_name="金卡_金",
            output_dir=output_dir,
        ).output()
        class_chain = NarrowFigureClassChain(
            stars_chain=stars_chain,
            class_list=[FG_DIR / CLASS_CONFIG["金卡"]["金"].format("Shielder")],
            class_name="金卡_金",
            output_dir=output_dir,
        ).output()

        print(f"\nNarrowFigure 链路: {len(figures)} 立绘 -> "
              f"frame {len(frame_chain)} -> border {len(border_chain)} -> "
              f"label {len(label_chain)} -> stars {len(stars_chain)} -> class {len(class_chain)}")
        print(f"  最终样例: {class_chain[0].image_path.name}")

        n = len(figures)
        self.assertEqual(len(frame_chain), n)
        self.assertEqual(len(border_chain), n * len(BORDER_CONFIG["金卡"]["金"]))
        self.assertEqual(len(label_chain), len(border_chain) * len(LABEL_LIST))
        self.assertEqual(len(stars_chain), len(label_chain) * len(STARS_CONFIG["金卡"]["金"]))
        self.assertEqual(len(class_chain), len(stars_chain))
        for node in class_chain:
            self.assertEqual(node.image.size, (170, 400))
            self.assertEqual(node.image.mode, "RGBA")

    def test_narrow_figure_frame_chain_with_mask(self):
        figures = NarrowFigureClassifyChain(RES_DIR / "assets" / "1").output()[:2]
        nodes = NarrowFigureFrameChain(
            name="masked",
            frame=FRAME_CONFIG["金卡"]["金"],
            narrow_figure_chain=figures,
            output_dir=TEMP_DIR / "support_test",
            mask=RES_DIR / "foreground_support" / "立绘掩码.png",
        ).output()

        print(f"\nNarrowFigureFrameChain (带立绘掩码) produced {len(nodes)}:")
        for node in nodes:
            print(f"  {node.image_path.name} size={node.image.size}")

        self.assertEqual(len(nodes), len(figures))
        for node in nodes:
            self.assertEqual(node.image.size, (170, 400))
            # 掩码切角：画布角落在掩码里是黑的，叠加后应保持透明
            self.assertEqual(node.image.getpixel((0, 0))[3], 0)

    def test_narrow_figure_craft_chain(self):
        output_dir = TEMP_DIR / "support_test"
        frame_chain = NarrowFigureFrameChain(
            name="金卡_金",
            frame=FRAME_CONFIG["金卡"]["金"],
            narrow_figure_chain=NarrowFigureClassifyChain(RES_DIR / "assets" / "1").output()[:2],
            output_dir=output_dir,
        ).output()
        cards = NarrowFigureClassChain(
            stars_chain=NarrowFigureStarsChain(
                label_chain=NarrowFigureLabelChain(
                    border_chain=NarrowFigureBorderChain(
                        frame_chain=frame_chain,
                        border_list=BORDER_CONFIG["金卡"]["金"][:1],
                        output_dir=output_dir,
                    ).output(),
                    label_list=LABEL_LIST[:1],
                    output_dir=output_dir,
                ).output(),
                stars_list=STARS_CONFIG["金卡"]["金"][:1],
                rank_name="金卡_金",
                output_dir=output_dir,
            ).output(),
            class_list=[FG_DIR / CLASS_CONFIG["金卡"]["金"].format("Shielder")],
            class_name="金卡_金",
            output_dir=output_dir,
        ).output()

        chain = NarrowFigureCraftChain(
            class_chain=cards,
            crop_mask=RES_DIR / "mask" / "礼装位置掩码.png",
            size=3,
            output_dir=output_dir,
        )
        nodes = chain.output()

        print(f"\nNarrowFigureCraftChain produced {len(nodes)} ({len(cards)} 卡面 × 3 背景):")
        for node in nodes[:3]:
            print(f"  {node.image_path.name} size={node.image.size}")

        self.assertEqual(len(nodes), len(cards) * 3)
        for idx, node in enumerate(nodes):
            card = cards[idx // 3]
            self.assertEqual(node.image.size, (170, 400))
            # 礼装窗 (11,245)-(160,315) 内被随机背景盖满，必须不透明
            self.assertEqual(node.image.getpixel((85, 280))[3], 255)
            # 窗口之外的卡面内容原样保留（取星级所在的一行）
            self.assertEqual(node.image.getpixel((85, 50)), card.image.getpixel((85, 50)))

    def test_classify_chains(self):
        input_dir = RES_DIR / "assets" / "1"
        
        # Test CharaClassifyChain
        chain = CharaClassifyChain(input_dir)
        nodes = chain.output()
        print(f"\nCharaClassifyChain found {len(nodes)} nodes:")
        for node in nodes:
            if hasattr(node, '_image_path') and node._image_path:
                print(f"  {node._image_path}")
        
        # Test FaceClassifyChain
        chain = FaceClassifyChain(input_dir)
        nodes = chain.output()
        print(f"\nFaceClassifyChain found {len(nodes)} nodes:")
        for node in nodes:
            if hasattr(node, '_image_path') and node._image_path:
                print(f"  {node._image_path}")
        
        # Test NarrowFigureClassifyChain
        chain = NarrowFigureClassifyChain(input_dir)
        nodes = chain.output()
        print(f"\nNarrowFigureClassifyChain found {len(nodes)} nodes:")
        for node in nodes:
            if hasattr(node, '_image_path') and node._image_path:
                print(f"  {node._image_path}")
        
        # Test CommandsClassifyChain
        chain = CommandsClassifyChain(input_dir)
        nodes = chain.output()
        print(f"\nCommandsClassifyChain found {len(nodes)} nodes:")
        for node in nodes:
            if hasattr(node, '_image_path') and node._image_path:
                print(f"  {node._image_path}")

        # Test StatusClassifyChain
        chain = StatusClassifyChain(input_dir)
        nodes = chain.output()
        print(f"\nStatusClassifyChain found {len(nodes)} nodes:")
        for node in nodes:
            if hasattr(node, '_image_path') and node._image_path:
                print(f"  {node._image_path}")

    def test_json_merger(self):
        json_files = sorted((RES_DIR / "chaldea").glob("servants.*.json"))
        merger = ChaldeaJsonMerger(json_files)
        merged = merger.merge()
        print(f"\nChaldeaJsonMerger: merged {len(json_files)} files -> {len(merged)} servants")
        self.assertGreater(len(merged), 0)
        self.assertIn(1, merged)

    def test_chara_face_pairer(self):
        json_files = sorted((RES_DIR / "chaldea").glob("servants.*.json"))
        merger = ChaldeaJsonMerger(json_files)
        merged = merger.merge()

        input_dir = RES_DIR / "assets" / "1"
        chara_chain = CharaClassifyChain(input_dir)
        face_chain = FaceClassifyChain(input_dir)
        pairer = CharaFacePairer(chara_chain, face_chain, merged)
        pairs = pairer.output()

        print(f"\nCharaFacePairer found {len(pairs)} pairs:")
        for chara, face in pairs:
            print(f"  {chara.image_path.name} <-> {face.image_path.name}")

        self.assertGreater(len(pairs), 0)
        for chara, face in pairs:
            self.assertIsNotNone(chara.image_path)
            self.assertIsNotNone(face.image_path)

    def test_avatar_detect_chain(self):
        json_files = sorted((RES_DIR / "chaldea").glob("servants.*.json"))
        merger = ChaldeaJsonMerger(json_files)
        merged = merger.merge()

        input_dir = RES_DIR / "assets" / "1"
        chara_chain = CharaClassifyChain(input_dir)
        face_chain = FaceClassifyChain(input_dir)
        pairer = CharaFacePairer(chara_chain, face_chain, merged)

        detect_chain = AvatarDetectChain(pairer)
        results = detect_chain.output()

        print(f"\nAvatarDetectChain found {len(results)} detections:")
        for r in results:
            print(f"  {r.chara_path.name} <- {r.face_path.name}")
            print(f"    score={r.score:.4f} scale={r.scale:.3f} rect={r.rect}")

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsNotNone(r.chara_path)
            self.assertIsNotNone(r.face_path)
            self.assertGreater(r.score, 0)
