import unittest

from scripts.gradio.output_filename import make_video_filename


class VideoOutputFilenameTest(unittest.TestCase):
    def test_distinct_long_prompts_do_not_share_a_filename(self):
        shared_prefix = "synthetic-prefix-0123456789012345"
        first = make_video_filename(f"{shared_prefix} left")
        second = make_video_filename(f"{shared_prefix} right")

        self.assertNotEqual(first, second)

    def test_repeated_prompt_gets_a_new_filename_for_each_generation(self):
        first = make_video_filename("synthetic repeated prompt")
        second = make_video_filename("synthetic repeated prompt")

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
