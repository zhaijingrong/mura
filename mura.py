import os
import tensorflow.compat.v2 as tf
import tensorflow_datasets.public_api as tfds

_DOWNLOAD_URL = "https://us13.mailchimp.com/mctx/clicks? \
                url=https%3A%2F%2Fcs.stanford.edu%2Fgroup%2Fmlgroup%2FMURA-v1.1.zip \
                &h=f27a1d45ea8264c93dd6cbafde0bb8694c0f9888d96ef5fcc68e8655a3cbb05c&v=1&xid=e5036c6a2a& \
                uid=55365305&pool=contact_facing&subject=MURA-v1.1%3A+Link+To+Dataset"

class MURA(tfds.core.GeneratorBasedBuilder):
    """ MURA dataset """

    VERSION = tfds.core.Version('0.1.0')

    def _info(self):
        # 指定 tfds.core.DatasetInfo对象
        return tfds.core.DatasetInfo(
            builder=self,
            # 这是将在数据集页面上显示的描述。
            description=("This is the dataset for MURA"),
            # tfds.features.FeatureConnectors
            features = tfds.features.FeaturesDict({
                "file_name": tfds.features.Text(),
                "extracted_dir_path": tfds.features.Text(),
            }),
        
            supervised_keys=("file_name", "extracted_dir_path"),
            # 用于文档的数据集主页
            homepage="https://www.github.com/zhaijingrong/mura",
            # 数据集的 Bibtex 引用
            citation=r"""@article{my-awesome-dataset-2020, author = {zhai, jingrong},"}""",
      )

    def _split_generators(self, dl_manager):
        # 下载源数据
        dl_path = dl_manager.download_and_extract(_DOWNLOAD_URL)
        extracted_path = os.path.join(dl_path, "MURA-v1.1")

        train_list = []
        val_list = []

        for root, _, filename in tf.io.gfile.walk(extracted_path):
            for fname in filename:
                full_file_name = os.path.join(root, fname)
                with tf.io.gfile.GFile(full_filename) as f:
                    for line in f:
                        if fname == "train_image_paths.csv":
                            train_list.append(line)
                        elif fname == "valid_image_paths.csv":
                            val_list.append(line)
      
        # 指定分割
        return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    gen_kwargs={
                        "file_name": train_list,
                        "extracted_dir_path": extracted_path,
                    },
                ),
                tfds.core.SplitGenerator(
                    name=tfds.Split.VALIDATION,
                    gen_kwargs={
                        "file_name": val_list,
                        "extracted_dir_path": extracted_path,
                    },
                ),
        ]

    def _generate_examples(self, file_name, extracted_dir_path):
        # 从数据集中产生样本
        # 从源文件中读取输入数据
        for iamge_name in file_name:
            yield iamge_name, {}


if __name__ == "__main__":
    mura = MURA()
