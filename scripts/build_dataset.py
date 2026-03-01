from src.data.sft_formater import SFTFormatter
from src.helper import config
from src.helper import prompts

def main():
    sft_formatter = SFTFormatter(config.TARGET_LANGUAGE, config.DATA_DIR, config.FILE_PATH_SAMPLE_STORY)
    sft_formatter.load_data()
    sft_formatter.format_data()
    sft_formatter.save_formatted_data(config.OUTPUT_DIR)