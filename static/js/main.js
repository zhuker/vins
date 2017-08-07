var initFileUploader = new Dropzone(".uploadform", { url: "/upload", previewsContainer: document.querySelector('.original')});
var categoriesDiv = document.querySelector('.categories')
initFileUploader.on('success', function(a,b) {
    console.log(a,b)
    b = JSON.parse(b)
    var original = document.querySelector('.original');
    original.innerHTML= '<img width=300 src="static/icons/'+b[0]+'">'
    var classes = b[1]
    categoriesDiv.innerHTML=''
    for (var i = 0; i < classes.length; i++) {
        var categoryDiv = document.createElement('div')
        categoryDiv.classList.add('category')
        categoryDiv.innerHTML = '<div class="catName">'+ classes[i][0] +'</div><div class="cat"><div class="catValue" style="width: '+ classes[i][1]*100 +'%"></div></div>'
        categoriesDiv.appendChild(categoryDiv)
    }
})


