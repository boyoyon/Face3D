<html lang="ja">
    <head>
        <meta charset="utf-8" />
    </head>
    <body>
        <h1><center>Face3D</center></h1>
        <h2>なにものか？</h2>
        <p>
            Mediapipeを使って取得したFacemeshをOpenGLで3D表示します。<br>
            <br>
            ※顔以外(首、髪の毛、背景など)は3D推定されていないので不自然に3D変形してしまいます。<br>
            "_(画像ファイル名).ply"に3Dメッシュを出力するので、必要に応じてBlenderなどで修正してください。<br>
            <img src="images/face3d.svg">
        </p>
        <h2>環境構築方法</h2>
        <p>
            pip install mediapipe PyOpenGL glfw
        </p>
        <h2>使い方</h2>
        <p>
            python Face3d.py (顔画像ファイル名)<br>
            <br>
            <table border="1">
                <tr><th>操作</th><th>機能</th></tr>
                <tr><td>左ボタン押下＋ドラッグ</td><td>3Dモデルの回転</td></tr>
                <tr><td>右ボタン押下＋ドラッグ</td><td>3Dモデルの移動</td></tr>
                <tr><td>ホイール回転</td><td>3Dモデルの拡大・縮小</td></tr>
                <tr><td>ホイールボタン押下</td><td>慣性モードのトグル(on⇔off)</td></tr>
                <tr><td>iキー押下</td><td>慣性モードのトグル(on⇔off)</td></tr>
                <tr><td>sキー押下</td><td>スクリーンショット保存</td></tr>
                <tr><td>ウィンドウ閉じるボタン押下　</td><td>プログラム終了</td></tr>
            </table>
        </p>
    </body>
</html>
