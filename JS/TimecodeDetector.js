var TimecodeDetector = function() {
    this.innerCanvas = document.createElement('canvas');

    this.haarCanvas = document.createElement('canvas');
    this.haarCanvas.width = 320;
    this.haarCanvas.height = 160;

    this.cropCanvas = document.createElement('canvas');
    this.cropCanvas.width = 189;
    this.cropCanvas.height = 27;

    this.haarDetector = new objectdetect.detector(this.haarCanvas.width, this.haarCanvas.height, 1.1, objectdetect.classifier);
    this.haarCorrector = new objectdetect.detector(this.cropCanvas.width, this.cropCanvas.height, 1.05, objectdetect.classifier);

    this.OCRModel = new KerasJS.Model({
      filepaths: {
        model: 'weights/OCRmodel.json',
        weights: 'weights/OCRmodel_weights_weights.buf',
        metadata: 'weights/OCRmodel_weights_metadata.json'
      },
      gpu: false
    });

    this.lastBbox = null;
};

TimecodeDetector.prototype._absToRel = function(rects, canvas) {
    for (let i = 0; i < rects.length; i++) {
        rects[i][0] /= canvas.width;
        rects[i][1] /= canvas.height;
        rects[i][2] /= canvas.width;
        rects[i][3] /= canvas.height
    }
    return rects
};

TimecodeDetector.prototype._relToAbs = function(rects, canvas) {
    for (let i = 0; i < rects.length; i++) {
        rects[i][0] = Math.round(rects[i][0] * canvas.width);
        rects[i][1] = Math.round(rects[i][1] * canvas.height);
        rects[i][2] = Math.round(Math.min(rects[i][2], 1) * canvas.width);
        rects[i][3] = Math.round(Math.min(rects[i][3], 1) * canvas.height)
    }
    return rects
};

TimecodeDetector.prototype._upscale = function(rects, wc = .1, hc = .1) {
    for (let i = 0; i < rects.length; i++) {
        rects[i][0] = Math.max(rects[i][0] - wc * rects[i][2] / 2, 0);
        rects[i][1] = Math.max(rects[i][1] - hc * rects[i][3] / 2, 0);
        rects[i][2] = rects[i][2] * (1 + wc);
        rects[i][3] = rects[i][3] * (1 + hc)
    }
    return rects
};

TimecodeDetector.prototype._magnify = function(rects) {
    let m_width = this.haarCanvas.width/2;
    for (let i = 0; i < rects.length; i++) {
        let cx = rects[i][0] + rects[i][2]/2, d = Math.abs(m_width - cx);
        if ((d / m_width) < 0.3) {
            rects[i][0] = m_width - rects[i][2]/2
        }
    }
    return rects
};

TimecodeDetector.prototype.detectBboxes = function(canvas) {
    canvas = canvas || this.haarCanvas;
    let rects = this.haarDetector.detect(canvas, 2);
    if (rects.lenght === 0) {
        console.log('nihuya')
    }

    rects = this._magnify(rects);
    rects = this._upscale(rects);

    if (this.lastBbox) {
        let b = this._relToAbs([this.lastBbox], canvas);
        rects = rects.concat(b)
    }
    rects = objectdetect.groupRectangles(rects, 1, 0.25);
    this.drawRects(rects, this.haarCanvas.getContext('2d'));
    rects = this._absToRel(rects, canvas);
    return rects
};

TimecodeDetector.prototype.drawRects = function (rects, ctx) {
    for (let i = 0; i < rects.length; ++i) {
        let coord = rects[i];
        ctx.beginPath();
        ctx.lineWidth = 1;
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.75)';
        ctx.rect(coord[0], coord[1], coord[2] * 1.2, coord[3] * 1.2);
        ctx.stroke();
    }
};

TimecodeDetector.prototype.readLabel = function(label){
    let vocab = '0123456789:;';
    function argmax(arr) {
        let maxId = -1, mval = -1;
        for (let i =0; i < vocab.length; i++) {
            if (arr[i] > mval) {
                maxId = i;
                mval = arr[i]
            }
        }
        return maxId
    }
    let tc = '';
    for (let i = 0; i < 11; i++) {
        tc += vocab[argmax(label.slice(i*12, (i+1)*12))]
    }
    return tc
};

TimecodeDetector.prototype.readBBox = function(bbox, callback) {
    let cropCoords = this._relToAbs([bbox], this.innerCanvas)[0],
        crop_ctx = this.cropCanvas.getContext('2d');

    crop_ctx.drawImage(this.innerCanvas, cropCoords[0], cropCoords[1], cropCoords[2], cropCoords[3],
        0, 0, this.cropCanvas.width, this.cropCanvas.height);

    let correctedBboxes = this.haarCorrector.detect(this.cropCanvas, 1);
    if (correctedBboxes.length) {
        correctedBboxes = this._upscale(correctedBboxes, .2);
        let m = 0, max_id = -1;
        for (let i = 0; i < correctedBboxes.length; i++) {
            let a = correctedBboxes[i][2]*correctedBboxes[i][3];
            if (m < a) {
                m = a;
                max_id = i;
            }
        }
        let b = correctedBboxes[max_id];
        crop_ctx.drawImage(this.cropCanvas, b[0], b[1], b[2], b[3],
            0, 0, this.cropCanvas.width, this.cropCanvas.height);

    }

    const imageData = crop_ctx.getImageData(0, 0, crop_ctx.canvas.width, crop_ctx.canvas.height);
    const {data, width, height } = imageData;

    let dataTensor = ndarray(new Float32Array(data), [width, height, 4]),
        dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3]);
    ndarray_ops.divseq(dataTensor, 255);
    ndarray_ops.subseq(dataTensor, 0.5);
    ndarray_ops.mulseq(dataTensor, 2);
    ndarray_ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0));
    ndarray_ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1));
    ndarray_ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2));

    const inputData = {input_1: dataProcessedTensor.data};

    this.OCRModel.predict(inputData).then(outputData => {
        let charProbs = outputData['chars'],
            isTimecode = outputData['conf'],
            res = {timecode: this.readLabel(charProbs, true), conf: isTimecode[0]};
        console.log(res);
        if (callback) {
            callback(res)
        }
    })
};


TimecodeDetector.prototype.getTimecode = function(img, callback) {
    let inner_ctx = this.innerCanvas.getContext('2d'),
        w = img.width || img.clientWidth,
        h = img.height || img.clientHeight;
    this.innerCanvas.width = w;
    this.innerCanvas.height = h;
    inner_ctx.drawImage(img, 0, 0, w, h, 0, 0, w, h);

    let haar_ctx = this.haarCanvas.getContext('2d');
    haar_ctx.drawImage(img, 0, 0, w, h, 0, 0, this.haarCanvas.width, this.haarCanvas.height);

    let bboxes = this.detectBboxes(this.haarCanvas),
        preds = [];

    let checkBox = function(self, currBBoxId) {
        self.readBBox(bboxes[currBBoxId], function (res) {
                preds.push(res);
                if (preds.length === bboxes.length) {
                    let maxConf = 0, maxId = 0;
                    for (let i = 0; i < preds.length; i++) {
                        if (preds[i].conf > maxConf) {
                            maxConf = preds[i].conf;
                            maxId = i
                        }
                    }

                    let bestResult = preds[maxId], timecode = '__:__:__:__';

                    if (bestResult.conf > 0.2) {
                        timecode = bestResult.timecode;
                        self.lastBbox = self._absToRel([bboxes[maxId]], self.innerCanvas)[0]
                    }

                    if (callback) {
                        callback(timecode)
                    }
                } else {
                    checkBox(self, currBBoxId+1)
                }
            })
    };
    checkBox(this, 0)
};

