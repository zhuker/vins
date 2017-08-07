/**
 * Created by oles on 04.06.16.
 */
PICTURES = []

var Picture = function(iconData) {
    this.id = iconData.id;
    this.imgUrl = iconData.imgUrl;
    this.categories = iconData.classes;
    this.reconstructionUrl = iconData.resUrl;
};

CURRENTIMAGE = {}

colorvocab = ["AliceBlue","AntiqueWhite","Aqua","Azure","Beige","Bisque","Black","BlanchedAlmond","Blue","BlueViolet","Brown","BurlyWood","CadetBlue","Chartreuse","Chocolate","Coral","CornflowerBlue","Cornsilk","Crimson","Cyan","DarkBlue","DarkCyan","DarkGoldenRod","DarkGray","DarkGrey","DarkGreen","DarkKhaki","DarkMagenta","DarkOliveGreen","Darkorange","DarkOrchid","DarkRed","DarkSalmon","DarkSeaGreen","DarkSlateBlue","DarkSlateGray","DarkSlateGrey","DarkTurquoise","DarkViolet","DeepPink","DeepSkyBlue","DimGray","DimGrey","DodgerBlue","FireBrick","FloralWhite","ForestGreen","Fuchsia","Gainsboro","GhostWhite","Gold","GoldenRod","Gray","Grey","Green","GreenYellow","HoneyDew","HotPink","IndianRed","Indigo","Ivory","Khaki","Lavender","LavenderBlush","LawnGreen","LemonChiffon","LightBlue","LightCoral","LightCyan","LightGoldenRodYellow","LightGray","LightGrey","LightGreen","LightPink","LightSalmon","LightSeaGreen","LightSkyBlue","LightSlateGray","LightSlateGrey","LightSteelBlue","LightYellow","Lime","LimeGreen","Linen","Magenta","Maroon","MediumAquaMarine","MediumBlue","MediumOrchid","MediumPurple","MediumSeaGreen","MediumSlateBlue","MediumSpringGreen","MediumTurquoise","MediumVioletRed","MidnightBlue","MintCream","MistyRose","Moccasin","NavajoWhite","Navy","OldLace","Olive","OliveDrab","Orange","OrangeRed","Orchid","PaleGoldenRod","PaleGreen","PaleTurquoise","PaleVioletRed","PapayaWhip","PeachPuff","Peru","Pink","Plum","PowderBlue","Purple","Red","RosyBrown","RoyalBlue","SaddleBrown","Salmon","SandyBrown","SeaGreen","SeaShell","Sienna","Silver","SkyBlue","SlateBlue","SlateGray","SlateGrey","Snow","SpringGreen","SteelBlue","Tan","Teal","Thistle","Tomato","Turquoise","Violet","Wheat","White","WhiteSmoke","Yellow","YellowGreen"];

var dropzone = new Dropzone('#uploaderBox', {
    url: "/upload",
    previewTemplate: document.querySelector('#preview-template').innerHTML,
    parallelUploads: 2,
    thumbnailHeight: null,
    thumbnailWidth: null,
    maxFilesize: 30,
    filesizeBase: 1000,
    thumbnail: function(file, dataUrl) {
      if (file.previewElement) {
        file.previewElement.classList.remove("dz-file-preview");
        var images = file.previewElement.querySelectorAll("[data-dz-thumbnail]");
        for (var i = 0; i < images.length; i++) {
          var thumbnailElement = images[i];
          thumbnailElement.alt = file.name;
          thumbnailElement.src = dataUrl;
          CURRENTIMAGE.src = dataUrl
        }
        setTimeout(function() { file.previewElement.classList.add("dz-image-preview"); }, 1);
      }
    }
});

var iconsTab = document.querySelector('.iconsTab')
var iconsCharts = document.querySelector('.iconsCharts')

var enableMode = function(viewMode) {
    var views = document.querySelectorAll('.viewMode')
    for (var i=0; i < views.length; i++) {
        views[i].classList.remove('current')
    }
    var activeView = document.querySelector('.' + viewMode)
    if (activeView) {
        activeView.classList.add('current')
    }
};

document.querySelector('.chartsLink').addEventListener('click', function(){
    enableMode('iconCharts')
})

document.querySelector('.uploadView').addEventListener('click', function(){
    enableMode('iconsUpload')
})

dropzone.on('success', function(a,pictureData) {
    result = JSON.parse(pictureData);
    var text = result.text
    console.log(text)
    var isNew = true
    id = Math.round(Math.random()*100000)
    iconsTab.innerHTML += '<li class=""><a href="#icon_'+id+'" data-toggle="tab" aria-expanded="false"><img width="40" src="'+ CURRENTIMAGE.src +'"></a></li>'
    if (PICTURES.length == 1) {
        iconsCharts.innerHTML += '<div class="tab-pane active" id="icon_'+id+'"></div>'
    } else {
        iconsCharts.innerHTML += '<div class="tab-pane" id="icon_'+id+'"></div>'
    }

    var newTab = iconsCharts.querySelector('#icon_'+id)
    newTab.style.font_size = '32px'
    currImg = document.createElement('img')
    currImg.src = result.originalImage
    currImg.style.width = '500px'
    currImg.style.margin = '20px'


    origImg = document.createElement('img')
    origImg.src = result.imgpath
    origImg.style.width = '500px'
    origImg.style.margin = '20px'

    newTab.innerHTML = '<div class="row"><div class="picturebox" id="picturebox_'+id+'"></div><table width="100%" class="icon-categories"></table><div class="results"></div></div>'
    pictureBox = newTab.querySelector('#picturebox_'+id)
    pictureBox.appendChild(origImg)
    pictureBox.appendChild(currImg)

    resdiv = pictureBox = newTab.querySelector('.results')


    var catTable = newTab.querySelector('.icon-categories')

    catTable.innerHTML=''
    var catRow = document.createElement('tr')
    catRow.innerHTML = '<td style="width:50%; font-size:42px">TEXT : </td><td style="width:50%; font-size:42px">'+ text +'</td></td>'
    catTable.appendChild(catRow)

    enableMode('iconCharts')
})
