package me.cgruber.dl.ocr

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.ProgramResult
import com.github.ajalt.clikt.core.main
import com.github.ajalt.clikt.core.subcommands
import com.github.ajalt.clikt.parameters.options.convert
import com.github.ajalt.clikt.parameters.options.default
import com.github.ajalt.clikt.parameters.options.option
import java.io.File
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import java.io.ByteArrayOutputStream
import java.io.PrintStream

class Ocr : CliktCommand(name = "ocr") {
  override fun run() {}
}

class Train : CliktCommand() {
  val dir by option("--dir", "-d", help = "Cache directory").convert { File(it) }.default(File("."))
  val modelFile by
    option("--model", "-m", help = "The model persistence file")
      .convert { File(it) }
      .default(File("models/my.model"))

  override fun run() {
    echo("Training model")
    val (training, test) = fashionMnist(dir)
    val model =
      Sequential.of(
        // comment to keep vertical style
        Input(28, 28, 1),
        Flatten(),
        Dense(100),
        Dense(50),
        Dense(10),
      )
    model.use {
      it.compile(
        optimizer = Adam(),
        loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
      )
      val buf = ByteArrayOutputStream()
      it.printSummary(PrintStream(buf))
      buf.flush()
      echo(buf.toString())

      // You can think of the training process as "fitting" the model to describe the given data :)
      it.fit(
        dataset = training,
        epochs = 10,
        batchSize = 100
      )

      val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

      echo("Accuracy: $accuracy")
      it.save(modelFile, writingMode = WritingMode.OVERRIDE)
    }
  }
}

class Run : CliktCommand() {
  override fun run() {
    echo("Loading testing data")
    val (_, test: OnHeapDataset) = fashionMnist()

    TensorFlowInferenceModel.load(File("model/my_model")).use {
      it.reshape(28, 28, 1)

      val prediction = it.predict(test.getX(0))
      val actualLabel = test.getY(0)

      println(
        "Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}."
      )
      println("Actual label is: $actualLabel.")
    }
  }
}

val stringLabels =
  mapOf(
    0 to "T-shirt/top",
    1 to "Trousers",
    2 to "Pullover",
    3 to "Dress",
    4 to "Coat",
    5 to "Sandals",
    6 to "Shirt",
    7 to "Sneakers",
    8 to "Bag",
    9 to "Ankle boots",
  )

fun main(vararg args: String) {
  Ocr().subcommands(Train(), Run()).main(args)
}
