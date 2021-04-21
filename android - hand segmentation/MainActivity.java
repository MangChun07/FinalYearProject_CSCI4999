package org.pytorch.imagesegmentation;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ProgressBar;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Map;

public class MainActivity extends AppCompatActivity implements Runnable {
    private ImageView mImageView;
    private Button mButtonSegment;
    private ProgressBar mProgressBar;
    private Bitmap mBitmap = null;
    private Module mModule = null;
    
    // modified
    private String mImagename = "my_hand_2.jpg";
    private Integer WIDTH = 225;
    private Integer HEIGHT = 400;

    // see http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/segexamples/index.html for the list of classes with indexes
    private static final int CLASSNUM = 2;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            Bitmap open = BitmapFactory.decodeStream(getAssets().open(mImagename));
            mBitmap = Bitmap.createScaledBitmap(open, WIDTH, HEIGHT, true);

        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }

        mImageView = findViewById(R.id.imageView);
        mImageView.setImageBitmap(mBitmap);

        final Button buttonRestart = findViewById(R.id.restartButton);
        buttonRestart.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
            	// modified
                try {
                    Bitmap open = BitmapFactory.decodeStream(getAssets().open(mImagename));
                    mBitmap = Bitmap.createScaledBitmap(open, WIDTH, HEIGHT, true);
                    mImageView.setImageBitmap(mBitmap);
                } catch (IOException e) {
                    Log.e("ImageSegmentation", "Error reading assets", e);
                    finish();
                }
            }
        });


        mButtonSegment = findViewById(R.id.segmentButton);
        mProgressBar = (ProgressBar) findViewById(R.id.progressBar);
        mButtonSegment.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                mButtonSegment.setEnabled(false);
                mProgressBar.setVisibility(ProgressBar.VISIBLE);
                mButtonSegment.setText(getString(R.string.run_model));

                Thread thread = new Thread(MainActivity.this);
                thread.start();
            }
        });

        try {
            mModule = Module.load(MainActivity.assetFilePath(getApplicationContext(), "TS_Handseg.pt"));
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }

    }

    @Override
    public void run() {
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        final Tensor outTensor = mModule.forward(IValue.from(inputTensor)).toTensor();
        final float[] outData = outTensor.getDataAsFloatArray();
        
        // modified
        int n = outData.length;
        float[] ypos = Arrays.copyOfRange(outData, 0, (n + 1)/2);
        float[] yneg = Arrays.copyOfRange(outData, (n + 1)/2, n);
        int[] intValues = new int[WIDTH * HEIGHT];
        for(int idx = 0; idx < (WIDTH * HEIGHT); idx++){
            if(ypos[idx] >= yneg[idx]){
                intValues[idx] = 0xFF000000;
            }
            else {
                intValues[idx] = 0xFFFFFFFF;
            }
        }

        Bitmap bmpSegmentation = Bitmap.createScaledBitmap(mBitmap, WIDTH, HEIGHT, true);
        Bitmap outputBitmap = bmpSegmentation.copy(bmpSegmentation.getConfig(), true);
        outputBitmap.setPixels(intValues, 0, outputBitmap.getWidth(), 0, 0, outputBitmap.getWidth(), outputBitmap.getHeight());
        final Bitmap transferredBitmap = Bitmap.createScaledBitmap(outputBitmap, mBitmap.getWidth(), mBitmap.getHeight(), true);

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mImageView.setImageBitmap(transferredBitmap);
                mButtonSegment.setEnabled(true);
                mButtonSegment.setText(getString(R.string.segment));
                mProgressBar.setVisibility(ProgressBar.INVISIBLE);

            }
        });
    }
}
