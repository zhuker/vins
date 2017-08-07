import tornado
import os
import json
import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, os.path, random, string
import tornado.httpserver
from tornado.options import define, options
import numpy as np
import logging
from nn.maskOcrModel import MaskOcrModel
reload(logging)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def load_img(path,target_size=None):
    from PIL import Image
    res = np.zeros((target_size[1], target_size[0], 3))
    try:
        img = Image.open(path)
        res = img.convert('RGB')
        if target_size:
            res = res.resize((target_size[1], target_size[0]))
    except:
        print('bad image file : '+str(path))
    return res

rootpath = os.path.dirname(os.path.abspath(__file__)).split('LALAFO')[0]+'LALAFO/'

define("port", default=4244, help="run on the given port", type=int)

curPath  = os.path.dirname(os.path.abspath(__file__))


OCR = MaskOcrModel()


class imageUploadHandler(tornado.web.RequestHandler):

    def post(self):
        file1 = self.request.files['file'][0]
        original_fname = file1['filename']
        extension = os.path.splitext(original_fname)[1]
        fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
        final_filename = fname+extension
        imgPath = curPath +"/static/input/" + final_filename
        output_file = open(imgPath, 'w')
        output_file.write(file1['body'])
        output_file.close()

        result = OCR.readText(imgPath)
        result['imgpath'] = result['imgpath'].replace(curPath, '')
        result['originalImage'] = imgPath.replace(curPath, '')
        self.write(json.dumps(result))
        self.finish()

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("static/index.html")

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/upload", imageUploadHandler),
            (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': curPath+'/static'}),
        ]
        tornado.web.Application.__init__(self, handlers)

def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
