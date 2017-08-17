resDivs = []


ndarray = require('ndarray')
ops = require('ndarray-ops')

function image2Darray(arr, width, height, rgb = [0, 0, 0]) {
  const size = width * height * 4
  let imageData = new Uint8ClampedArray(size)
  for (let i = 0; i < size; i += 4) {
    imageData[i] = rgb[0]
    imageData[i + 1] = rgb[1]
    imageData[i + 2] = rgb[2]
    imageData[i + 3] = 255 * arr[i / 4]
  }
  return new ImageData(imageData, width, height)
}

function addNewImage(i) {
    if (i < 32) {

        //////////////////////////////////////////////////////////////
        imgDiv = document.createElement('div')
        imgDiv.id = 'img_'+i
        imgDiv.class = 'imgresults'

        img_crop_canvas = document.createElement('canvas')
        img_crop_canvas.id = 'img_crop_'+i
        img_crop_canvas.width=512
        img_crop_canvas.height=42

        img_mask_canvas = document.createElement('canvas')
        img_mask_canvas.id = 'img_mask_'+i
        img_mask_canvas.width=500
        img_mask_canvas.height=30

        tModel.crop_canvas = img_crop_canvas
        tModel.mask_canvas = img_mask_canvas

        nextImg = document.createElement('img')
        nextImg.imgId = i
        nextImg.src = 'imgs/'+i+'.jpg'
        nextImg.onload = function (e) {
            currentImgId= this.imgId
            tModel.originalImage = nextImg
            tModel.loadImageToCanvas('imgs/'+i+'.jpg', function (timecode) {
                imgDiv.innerHTML += '<p>' + timecode + '</p></br>'
                imgDiv.appendChild(nextImg)
                imgDiv.appendChild(tModel.crop_canvas)
                imgDiv.appendChild(tModel.mask_canvas)
                resultDiv = document.querySelector('#results')
                resultDiv.appendChild(imgDiv)
                resDivs.push(imgDiv)
                addNewImage(i+1)
                // Tesseract.recognize(tModel.crop_canvas, {
                // lang: 'eng',
                // tessedit_char_whitelist: '1234567890:; ',//'abcdefghijklmnopqrstuvwxyzQWERTYUIOPASDFGHJKLZXCVBNM-_‘»~“?=‘€€¢—ﬁ£¥”}ﬂ↵{][^%’®$§#@!&*()-+/<>,.\`\'\"\\\|'
                // })
                // .then(function(result){
                //     console.log(result)
                //     imgDiv.innerHTML += '<p>' + result.text + '</p></br>'
                //     imgDiv.appendChild(nextImg)
                //     imgDiv.appendChild(tModel.crop_canvas)
                //     imgDiv.appendChild(tModel.mask_canvas)
                //     resultDiv = document.querySelector('#results')
                //     resultDiv.appendChild(imgDiv)
                //     resDivs.push(imgDiv)
                //     addNewImage(i+1)
                // })

            })
        }
    }

}


function image2Darray(arr, width, height, rgb = [0, 0, 0]) {
  const size = width * height * 4
  let imageData = new Uint8ClampedArray(size)
  for (let i = 0; i < size; i += 4) {
    imageData[i] = rgb[0]
    imageData[i + 1] = rgb[1]
    imageData[i + 2] = rgb[2]
    imageData[i + 3] = 255 * arr[i / 4]
  }
  return new ImageData(imageData, width, height)
}

var JSTimecodes = function(){
    this.imageLoading = false;
    this.imageLoadingError = false;
    this.originalImage = false;
    this.callback = false;
    this.bboxModel = new KerasJS.Model({
      filepaths: {
        model: 'weights/bbModel.json',
        weights: 'weights/bbModel_weights_weights.buf',
        metadata: 'weights/bbModel_weights_metadata.json'
      },
      gpu: false
    })
    this.segModel = new KerasJS.Model({
      filepaths: {
        model: 'weights/cropSegmodel.json',
        weights: 'weights/cropSegmodel_weights_weights.buf',
        metadata: 'weights/cropSegmodel_weights_metadata.json'
      },
      gpu: false
    })
    this.OCRModel = new KerasJS.Model({
      filepaths: {
        model: 'weights/OCRmodel.json',
        weights: 'weights/OCRmodel_weights_weights.buf',
        metadata: 'weights/OCRmodel_weights_metadata.json'
      },
      gpu: false
    })

    this.loadImageToCanvas = function(url, callback) {
      if (!url) {
        this.clearAll()
        return
      }

      if (callback) {
          this.callback = callback
      }
      this.imageLoading = true
      loadImage(
        url,
        img => {
          if (img.type === 'error') {
            this.imageLoadingError = true
            this.imageLoading = false
          } else {
            // load image data onto input canvas
            canv = document.getElementById('input-canvas')
            const ctx = canv.getContext('2d')
            ctx.drawImage(img, 0, 0, img.width,    img.height,     // source rectangle
                   0, 0, canv.width, canv.height);
            detector = HAARDetector(canv)
            detector.detectBboxes()
            this.imageLoadingError = false
            this.imageLoading = false
            this.modelRunning = true
            if (this.callback) {
                this.callback('ololo')
            }
            // model predict
            // setTimeout(() => {
            //     this.runBBoxModel()
            //   }, 200)
          }
        },
        { maxWidth: 320, maxHeight: 160, cover: true, crop: true, canvas: true, crossOrigin: 'Anonymous' }
      )
    }

    this.runBBoxModel = function() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.divseq(dataTensor, 255)
      ops.subseq(dataTensor, 0.5)
      ops.mulseq(dataTensor, 2)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))

      const inputData = { input_2: dataProcessedTensor.data }
      this.bboxModel.predict(inputData).then(outputData => {
        this.output = outputData['bbox_out']
        this.modelRunning = false
        this.crop(outputData['bbox_out'])

        this.runOCRModel()

        //this.runSegModel()
        //this.drawOutput(outputData['conv2d_4'])
      })
    }


    this.readLabel = function(label){
        var vocab = '0123456789:;'
        function argmax(arr) {
            maxId = -1
            mval = -1
            for (var i =0; i < arr.length; i++) {
                if (arr[i] > mval) {
                    maxId = i
                    mval = arr[i]
                }
            }
            return maxId
        }
        var tc = ''
        for (var i =0; i < 11; i++) {
            tc += vocab[argmax(label.slice(i*12,(i+1)*12))]
        }
        return tc
    }

    this.runOCRModel = function() {
      const ctx = tModel.crop_canvas.getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.divseq(dataTensor, 255)
      ops.subseq(dataTensor, 0.5)
      ops.mulseq(dataTensor, 2)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))

      const inputData = { input_1: dataProcessedTensor.data }
      this.OCRModel.predict(inputData).then(outputData => {
        this.ocrOutput = outputData['activation_1']
        var timecode = this.readLabel(outputData['activation_1'])
        if (this.callback) {
            this.callback(timecode)
        }
      })
    }

    this.runSegModel = function() {
      const ctx = tModel.crop_canvas.getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
      ops.divseq(dataTensor, 255)
      ops.subseq(dataTensor, 0.5)
      ops.mulseq(dataTensor, 2)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))

      const inputData = { input_1: dataProcessedTensor.data }
      this.segModel.predict(inputData).then(outputData => {
        this.output = outputData['conv2d_5']
        this.modelRunning = false
        this.drawOutput(outputData['conv2d_5'])
        if (this.callback) {
            this.callback()
        }
      })
    }

    this.crop = function (bbox) {
        console.log(bbox)
        img = this.originalImage
        rel_width = bbox[2]*1.6
        rel_height = bbox[3]*1.2
        tl_x = Math.max(((bbox[0]-(rel_width/2))*img.width),0)
        tl_y = Math.max(((bbox[1]-(rel_height/2))*img.height),0)

        cropWidth = Math.floor(rel_width*img.width)
        cropHeight =  Math.floor(rel_height*img.height)

        outcanvas = tModel.crop_canvas
        outcanvas.width = 512
        outcanvas.height = 32
        ctx = outcanvas.getContext('2d')

        ctx.drawImage(img, tl_x, tl_y, cropWidth, cropHeight, 0, 0, 512, 32);

        return ctx
    }

    this.drawOutput = function(output) {
      const ctx = tModel.mask_canvas.getContext('2d')
      const image = image2Darray(output, 512, 32, [0, 0, 0])
      ctx.putImageData(image, 0, 0)
    }

    return this
}


var HAARDetector = function(canvas){
    this.canvas = canvas
    this.classifier = objectdetect.classifier;

    this.detectBboxes = function() {
		var context = this.canvas.getContext('2d');
		this.detector = new objectdetect.detector(this.canvas.width, this.canvas.height, 1.1, this.classifier);
        var rects = this.detector.detect(canvas, 2);
        rects = this.magnify(rects)
        rects = this.objectdetect.groupRectangles(rects, 1)
        this.drawRects(rects,context)
    }

    this.drawRects = function(rects, ctx) {
        for (var i = 0; i < rects.length; ++i) {
            var coord = rects[i];
            ctx.beginPath();
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(0, 255, 255, 0.75)';
            ctx.rect(coord[0], coord[1], coord[2]*1.2, coord[3]*1.2);
            ctx.stroke();
        }
    }
    this.magnify = function(rects) {
        var m_width = this.canvas.width/2
        for (var i=0; i<rects.length; i++) {
            var cx = rects[i][0] + rects[i][2]/2
            var d = Math.abs(m_width - cx)
            if ((d / m_width) < 0.3) {
                rects[i][0] = m_width - rects[i][2]/2 - (0.05 * m_width)
            }
        }
        return rects
    }
    return this
}

tModel = JSTimecodes()
tModel.bboxModel.ready().then(() => {
        addNewImage(1)
    })