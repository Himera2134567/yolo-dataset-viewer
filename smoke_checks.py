import hashlib
import io
import tempfile
import unittest
from pathlib import Path


def make_suite(module):
    class ViewerChecks(unittest.TestCase):
        def setUp(self):
            self.temp = tempfile.TemporaryDirectory()
            self.addCleanup(self.temp.cleanup)
            self.labels = Path(self.temp.name)/'sample.txt'

        def test_boxes_and_polygons(self):
            self.labels.write_text('2 .5 .5 .4 .2\n1 .1 .1 .9 .1 .5 .9\n', encoding='utf-8-sig')
            shapes = module.YoloSegViewerApp._parse_yolo_labels(str(self.labels), 100, 50)
            self.assertEqual([s['type'] for s in shapes], ['rect','polygon'])
            self.assertEqual(shapes[0]['class_id'], 2)
            self.assertEqual(shapes[0]['points'], [(30.0,20.0),(70.0,30.0)])
            self.assertEqual(len(shapes[1]['points']), 3)

        def test_malformed_rows_are_skipped(self):
            self.labels.write_text('x .5 .5 .2 .2\n1.5 .5 .5 .2 .2\n0 nan .5 .2 .2\n0 .5 .5 -.2 .2\n0 2 .5 .2 .2\n0 .1 .1 .2 .2 .3\n0 .5 .5 .2 .2\n', encoding='utf-8')
            self.assertEqual(len(module.YoloSegViewerApp._parse_yolo_labels(str(self.labels),100,50)),1)

        def test_missing_and_clipped_labels(self):
            self.assertEqual(module.YoloSegViewerApp._parse_yolo_labels(str(self.labels),100,50), [])
            self.labels.write_text('0 0 .5 .4 .2', encoding='utf-8')
            shape = module.YoloSegViewerApp._parse_yolo_labels(str(self.labels),100,50)[0]
            self.assertEqual(shape['points'][0][0],0)
            self.assertEqual(shape['points'][1][0],20)

        def test_scale_preserves_ratio(self):
            image = module.Image.new('RGB',(100,50))
            self.assertEqual(module.YoloSegViewerApp._scale_to_fit(image,20,20).size,(20,10))
            self.assertIs(module.YoloSegViewerApp._scale_to_fit(image,200,200),image)

        def test_overlay_does_not_modify_original(self):
            image = module.Image.new('RGB',(50,50),'black')
            rendered = module.YoloSegViewerApp._draw_shapes(image,[{'type':'rect','points':[(5,5),(20,20)]}])
            self.assertEqual(image.getpixel((10,10)),(0,0,0))
            self.assertNotEqual(rendered.getpixel((10,10)),(0,0,0))
    return unittest.defaultTestLoader.loadTestsFromTestCase(ViewerChecks)


def run_checks(module):
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output).run(make_suite(module))
    if not result.wasSuccessful():
        raise RuntimeError(output.getvalue())
    with tempfile.TemporaryDirectory() as directory:
        base=Path(directory)
        (base/'images').mkdir(); (base/'labels').mkdir()
        module.Image.new('RGB',(80,40),'navy').save(base/'images/a.png')
        module.Image.new('RGB',(80,40),'green').save(base/'images/b.JPG')
        (base/'labels/a.txt').write_text('0 .5 .5 .5 .5\n',encoding='utf-8')
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in base.rglob('*') if p.is_file()}
        root=module.tk.Tk(); root.withdraw()
        try:
            app=module.YoloSegViewerApp(root)
            app.load_dataset(str(base))
            self_photo=app.current_photo
            assert self_photo is not None and len(app.image_paths)==2
            app.show_next(); assert app.current_index==1
            app.show_next(); assert app.current_index==0
            app.show_prev(); assert app.current_index==1
            root.update_idletasks()
            if app._resize_job is not None:root.after_cancel(app._resize_job)
        finally:root.destroy()
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    return {'ok':True,'tests_passed':result.testsRun,'gui_navigation_and_images':'passed','dataset_files_unchanged':True,'private_data_read':False}
