import unittest

from PIL import Image

from chaldea.chain import (
    ChainNode,
    CharaClassifyChain,
    FaceClassifyChain,
    NarrowFigureClassifyChain,
    CommandsClassifyChain,
    ChaldeaJsonMerger,
    CharaFacePairer,
    AvatarDetectChain,
    ExpandClipChain,
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
            mask_img=RES_DIR / "mask" / "BGO头像裁剪掩码.png",
        )
        nodes = chain.output()

        print(f"\nExpandClipChain produced {len(nodes)} clips:")
        for node in nodes:
            print(f"  {node.image_path} size={node.image.size}")

        self.assertGreater(len(nodes), 0)
        self.assertLessEqual(len(nodes), 6)
        with Image.open(RES_DIR / "mask" / "BGO头像裁剪掩码.png") as mask:
            mask_size = mask.size
        for node in nodes:
            self.assertTrue(node.image_path.exists())
            self.assertEqual(node.image.size, mask_size)
            self.assertEqual(node.image.mode, "RGBA")

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
