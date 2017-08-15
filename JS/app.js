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
            tModel.loadImageToCanvas('imgs/'+i+'.jpg', function () {
                 Tesseract.recognize(tModel.crop_canvas, {
                lang: 'eng',
                tessedit_char_whitelist: '1234567890:; ',//'abcdefghijklmnopqrstuvwxyzQWERTYUIOPASDFGHJKLZXCVBNM-_‘»~“?=‘€€¢—ﬁ£¥”}ﬂ↵{][^%’®$§#@!&*()-+/<>,.\`\'\"\\\|'
                })
                .then(function(result){
                    console.log(result)
                    imgDiv.innerHTML += '<p>' + result.text + '</p></br>'
                    imgDiv.appendChild(nextImg)
                    imgDiv.appendChild(tModel.crop_canvas)
                    imgDiv.appendChild(tModel.mask_canvas)
                    resultDiv = document.querySelector('#results')
                    resultDiv.appendChild(imgDiv)
                    resDivs.push(imgDiv)
                    addNewImage(i+1)
                })
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
            this.imageLoadingError = false
            this.imageLoading = false
            this.modelRunning = true
            // model predict
            setTimeout(() => {
                this.runBBoxModel()
              }, 200)
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

      const inputData = { input_1: dataProcessedTensor.data }
      this.bboxModel.predict(inputData).then(outputData => {
        this.output = outputData['bbox_out']
        this.modelRunning = false
        this.crop(outputData['bbox_out'])

        if (this.callback) {
            this.callback()
        }

        //this.runSegModel()
        //this.drawOutput(outputData['conv2d_4'])
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
        rel_width = bbox[2]*2
        rel_height = bbox[3]*1.9
        tl_x = Math.max(((bbox[0]-(rel_width/2))*img.width),0)
        tl_y = Math.max(((bbox[1]-(rel_height/2))*img.height),0)

        cropWidth = Math.floor(rel_width*img.width)
        cropHeight =  Math.floor(rel_height*img.height)

        outcanvas = tModel.crop_canvas
        outcanvas.width = 512
        outcanvas.height = 42
        ctx = outcanvas.getContext('2d')

        ctx.drawImage(img, tl_x, tl_y, cropWidth, cropHeight, 0, 0, 512, 42);

        return ctx
    }

    this.drawOutput = function(output) {
      const ctx = tModel.mask_canvas.getContext('2d')
      const image = image2Darray(output, 500, 30, [0, 0, 0])
      ctx.putImageData(image, 0, 0)
    }

    return this
}


tModel = JSTimecodes()
tModel.bboxModel.ready().then(() => {
        addNewImage(1)
    })