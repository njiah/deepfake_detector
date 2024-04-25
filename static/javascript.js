const dropArea = document.getElementById('drop-area');
const image = document.getElementById('image');
const imageView = document.getElementById('img-view');

image.addEventListener('change', uploadImage);

function uploadImage() {
    let imgLink = URL.createObjectURL(image.files[0]);
    imageView.style.backgroundImage = `url(${imgLink})`;
    imageView.textContent = '';
    imageView.style.border = 0;
}

dropArea.addEventListener('dragover', function(e){
    e.preventDefault();
});

dropArea.addEventListener('drop', function(e){
    e.preventDefault();
    image.files = e.dataTransfer.files;
    uploadImage();
});

/*document.querySelectorAll('.custom-file_input').forEach((inputElement) => {
    const dropZoneElement = inputElement.closest('.custom-file');
    dropZoneElement.addEventListener('click', (e) => {
        inputElement.click();
    });
    inputElement.addEventListener('change', (e) => {
        if (inputElement.files.length) {
            updateThumbnail(dropZoneElement, inputElement.files[0]);
        }
    });
    dropZoneElement.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZoneElement.classList.add('custom-file_dragover');
    });
    ['dragleave', 'dragend'].forEach((type) => {
        dropZoneElement.addEventListener(type, (e) => {
            dropZoneElement.classList.remove('custom-file_dragover');
        });
    });
    dropZoneElement.addEventListener('drop', (e) => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            inputElement.files = e.dataTransfer.files;
            updateThumbnail(dropZoneElement, e.dataTransfer.files[0]);
        }
        dropZoneElement.classList.remove('custom-file_dragover');
    });
});
    function updateThumbnail(dropZoneElement, file) {
        let thumbnailElement = dropZoneElement.querySelector('.custom-file_thumb');
        if (dropZoneElement.querySelector('.custom-file_prompt')) {
            dropZoneElement.querySelector('.custom-file_prompt').remove();
        }
        thumbnailElement.dataset.label = file.name;
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                thumbnailElement.style.backgroundImage = `url('${reader.result}')`;
            };
        }
        else {
            thumbnailElement.style.backgroundImage = null;
        }
    }
*/