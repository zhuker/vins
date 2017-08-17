var TimecodeDetector = function() {
    this.innerCanvas = document.createElement('canvas')

    this.haarCanvas = document.createElement('canvas')
    this.haarCanvas.width = 320
    this.haarCanvas.height = 160

    this.cropCanvas = document.createElement('canvas')
    this.cropCanvas.width = 189
    this.cropCanvas.height = 27

    this.haarDetector = new objectdetect.detector(this.haarCanvas.width, this.haarCanvas.height, 1.1, objectdetect.classifier);

    this.OCRModel = new KerasJS.Model({
      filepaths: {
        model: 'weights/OCRmodel.json',
        weights: 'weights/OCRmodel_weights_weights.buf',
        metadata: 'weights/OCRmodel_weights_metadata.json'
      },
      gpu: false
    })
}

TimecodeDetector.prototype._absToRel = function(rects, canvas) {
    for (var i=0; i<rects.length; i++) {
        rects[i][0] /= canvas.width
        rects[i][1] /= canvas.height
        rects[i][2] /= canvas.width
        rects[i][3] /= canvas.height
    }
    return rects
}

TimecodeDetector.prototype._relToAbs = function(rects, canvas) {
    for (var i=0; i<rects.length; i++) {
        rects[i][0] = Math.round(rects[i][0] * canvas.width)
        rects[i][1] = Math.round(rects[i][1] * canvas.height)
        rects[i][2] = Math.round(rects[i][2] * canvas.width)
        rects[i][3] = Math.round(rects[i][3] * canvas.height)
    }
    return rects
}

TimecodeDetector.prototype._magnify = function(rects) {
    var m_width = this.haarCanvas.width/2;
    for (var i=0; i<rects.length; i++) {
        var cx = rects[i][0] + rects[i][2]/2
        var d = Math.abs(m_width - cx)
        if ((d / m_width) < 0.3) {
            rects[i][0] = m_width - rects[i][2]/2
        }
    }
    return rects
}

TimecodeDetector.prototype.detectBboxes = function(canvas) {
    canvas = canvas || this.haarCanvas
    var rects = this.haarDetector.detect(canvas, 2);
    rects = this._magnify(rects)
    rects = objectdetect.groupRectangles(rects, 1)
    rects = this._absToRel(rects, canvas)
    return rects
}

TimecodeDetector.prototype.readLabel = function(label, isTimecode){
        var vocab = '0123456789:;'
        function argmax(arr) {
            maxId = -1
            mval = -1
            for (var i =0; i < vocab.length; i++) {
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

TimecodeDetector.prototype.readBBox = function(bbox, callback) {
    var cropCoords = this._relToAbs([bbox], this.innerCanvas)[0]
    var crop_ctx = this.cropCanvas.getContext('2d');

    crop_ctx.drawImage(this.innerCanvas, cropCoords[0], cropCoords[1], cropCoords[2], cropCoords[3],
        0, 0, this.cropCanvas.width, this.cropCanvas.height);

    const imageData = crop_ctx.getImageData(0, 0, crop_ctx.canvas.width, crop_ctx.canvas.height)
    const { data, width, height } = imageData

    var dataTensor = ndarray(new Float32Array(data), [width, height, 4])
    var dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])
    ndarray_ops.divseq(dataTensor, 255)
    ndarray_ops.subseq(dataTensor, 0.5)
    ndarray_ops.mulseq(dataTensor, 2)
    ndarray_ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
    ndarray_ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
    ndarray_ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))

    const inputData = { input_1: dataProcessedTensor.data }

    this.OCRModel.predict(inputData).then(outputData => {
        var charProbs = outputData['activation_1']
        var isTimecode = outputData['dense_1']
        var res = {timecode: this.readLabel(charProbs, true), conf: isTimecode[0]}
        console.log(res)
        if (callback) {
            callback(res)
        }
    })
}


TimecodeDetector.prototype.getTimecode = function(img, callback) {
    var inner_ctx = this.innerCanvas.getContext('2d');
    inner_ctx.drawImage(img, 0, 0, img.width,    img.height,
                   0, 0, this.innerCanvas.width, this.innerCanvas.height);

    var haar_ctx = this.haarCanvas.getContext('2d');
    haar_ctx.drawImage(img, 0, 0, img.width,    img.height,
                   0, 0, this.haarCanvas.width, this.haarCanvas.height);

    var bboxes = this.detectBboxes(this.haarCanvas)
    var preds = []

    var checkBox = function(self, currBBoxId) {
        self.readBBox(bboxes[currBBoxId], function (res) {
                preds.push(res)
                predicted = true
                if (preds.length == bboxes.length) {
                    var maxConf = 0
                    var maxId = 0
                    for (var i=0; i < preds.length; i++) {
                        if (preds[i].conf > maxConf) {
                            maxConf = preds[i].conf
                            maxId = i
                        }
                    }

                    var bestResult = preds[maxId]
                    var timecode = '__:__:__:__'

                    if (bestResult.conf > 0.5) {
                        timecode = bestResult.timecode
                    }

                    if (callback) {
                        callback(timecode)
                    }
                } else {
                    checkBox(self, currBBoxId+1)
                }
            })
    }
    checkBox(this, 0)
}

