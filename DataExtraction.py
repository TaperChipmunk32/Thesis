'''
File for extracting individual verses from the XML files and storing them in a dictionary.
Creates a dictionary with the following structure:
{
    'translation_name': {
        'book_name': {
            'chapter_num': {
                'verse_num': 'verse_text'
            }
        }
    }
'''
import xml.etree.ElementTree as ET
import os

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    data = {}
    for book in root.findall('.//b'):
        book_name = book.get('n')
        if book_name in ['Matthew', 'Mark', 'Luke', 'John']:
            data[book_name] = {}
            for chapter in book.findall('.//c'):
                chapter_num = int(chapter.get('n'))
                data[book_name][chapter_num] = {}
                for verse in chapter.findall('.//v'):
                    verse_num = int(verse.get('n'))
                    data[book_name][chapter_num][verse_num] = verse.text
    return data

def extract_data():
    translations_dir = 'translations'
    data = {}
    for file_name in os.listdir(translations_dir):
        if file_name.endswith('.xml'):
            file_path = os.path.join(translations_dir, file_name)
            translation_name = os.path.splitext(file_name)[0]
            data[translation_name] = parse_xml(file_path)
    return data