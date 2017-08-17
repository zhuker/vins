function loadImage(url, callback) {
    var img = document.createElement('img');
    img.src = url;
    img.onload = function (e) {
        callback(img)
    }
}

Detector = new TimecodeDetector()
var N=1;

var processInput = function(img) {
    Detector.getTimecode(img, function(timecode) {
        currentTimecode.innerHTML = JSON.stringify(timecode)
        N += 1
        loadImage('imgs/'+N+'.jpg', processInput)
    })
}


Detector.OCRModel.ready().then(() => {
    document.body.appendChild(Detector.innerCanvas)
    document.body.appendChild(Detector.haarCanvas)
    document.body.appendChild(Detector.cropCanvas)

    currentTimecode = document.createElement('div')
    document.body.appendChild(currentTimecode)
    loadImage('imgs/'+N+'.jpg', processInput)
})
