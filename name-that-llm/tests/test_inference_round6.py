import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import inference_round6 as inference


class InferenceContractTests(unittest.TestCase):
    def test_test_schema_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.csv'
            p.write_text('id,text\na,hello\nb,world\n',encoding='utf-8')
            self.assertEqual(inference.read_test(p),(['a','b'],['hello','world']))
            p.write_text('id,text\na,hello\na,world\n',encoding='utf-8')
            with self.assertRaises(ValueError): inference.read_test(p)
            p.write_text('id,text,model\na,hello,hidden\n',encoding='utf-8')
            with self.assertRaises(ValueError): inference.read_test(p)

    def test_row_limit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.csv'
            p.write_text('id,text\n'+''.join(f'{i},synthetic\n' for i in range(4096)),encoding='utf-8')
            self.assertEqual(len(inference.read_test(p)[0]),4096)
            with p.open('a') as f: f.write('4096,synthetic\n')
            with self.assertRaises(ValueError): inference.read_test(p)

    def test_class_mapping_and_output_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'submission.parquet'
            labels=[f'label-{i}' for i in range(18)]
            probability=np.zeros((3,18)); probability[np.arange(3),[17,0,5]]=1
            inference.write_submission(p,['z','a','q'],probability,labels)
            import pyarrow.parquet as pq
            self.assertEqual(pq.read_table(p).to_pydict(),{'id':['z','a','q'],'model':['label-17','label-0','label-5']})
            with self.assertRaises(FileExistsError): inference.write_submission(p,['z','a','q'],probability,labels)


if __name__=='__main__': unittest.main()
