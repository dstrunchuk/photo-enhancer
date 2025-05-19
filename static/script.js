async function uploadPhoto() {
    const input = document.getElementById("fileInput");
    const loading = document.getElementById("loading");
    const preview = document.getElementById("preview");

    if (!input.files[0]) {
        alert("Выберите изображение!");
        return;
    }

    const formData = new FormData();
    formData.append("file", input.files[0]);

    loading.style.display = "block";
    preview.style.display = "none";

    try {
        const response = await fetch("/upload/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Ошибка сервера");

        const blob = await response.blob();
        const imgUrl = URL.createObjectURL(blob);

        preview.src = imgUrl;
        preview.style.display = "block";
    } catch (error) {
        alert("Произошла ошибка: " + error.message);
    } finally {
        loading.style.display = "none";
    }
}