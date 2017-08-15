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

var JSTimecodes = function(){
    this.imageLoading = false;
    this.imageLoadingError = false;
    this.segmentModel = new KerasJS.Model({
      filepaths: {
        model: 'weights/segmodel.json',
        weights: 'weights/segmodel_weights_weights.buf',
        metadata: 'weights/segmodel_weights_metadata.json'
      },
      gpu: true
    })

    this.loadImageToCanvas = function(url) {
      if (!url) {
        this.clearAll()
        return
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
            const ctx = document.getElementById('input-canvas').getContext('2d')
            ctx.drawImage(img, 0, 0)
            this.imageLoadingError = false
            this.imageLoading = false
            this.modelRunning = true
            // model predict
            setTimeout(() => {
                this.runModel()
              }, 200)
          }
        },
        { maxWidth: 320, maxHeight: 160, cover: true, crop: true, canvas: true, crossOrigin: 'Anonymous' }
      )
    }

    this.runModel = function() {
      const ctx = document.getElementById('input-canvas').getContext('2d')
      const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height)
      const { data, width, height } = imageData

      let dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      let dataProcessedTensor = ndarray(new Float32Array(width * height * 1), [width, height, 1])
      ops.divseq(dataTensor, 255)
      ops.subseq(dataTensor, 0.5)
      ops.mulseq(dataTensor, 2)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 1))

      const inputData = { input_1: dataProcessedTensor.data }
      this.segmentModel.predict(inputData).then(outputData => {
        this.output = outputData['conv2d_7']
        this.modelRunning = false
        this.drawOutput(outputData['conv2d_7'])
      })
    }

    this.drawOutput = function(output) {
      const ctx = document.getElementById('output-canvas').getContext('2d')
      const image = image2Darray(output, 320, 160, [0, 0, 70])
      ctx.putImageData(image, 0, 0)
    }

    return this
}


tModel = JSTimecodes()
tModel.segmentModel.ready().then(() => {
        tModel.loadImageToCanvas('imgs/8.jpg')
    })